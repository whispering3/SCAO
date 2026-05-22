"""
Colab T4 smoke test for SCAO.

Run from the repository root:

    python scripts/colab_t4_smoke_test.py

Optional:

    python scripts/colab_t4_smoke_test.py --compile-cuda-ext
    python scripts/colab_t4_smoke_test.py --quick

The script checks:
  - CUDA availability and GPU name
  - optional CUDA extension build/load
  - fused preconditioner identity case
  - SCAO async_precond training smoke test
  - scao_1b preset training smoke test on a small TransformerEncoder
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import time
import warnings
from pathlib import Path

import torch
import torch.nn as nn

from scao import SCAO, scao_1b
from scao.cuda import fused_kronecker_precond

ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd or ROOT), check=True)


def _require_cuda() -> torch.device:
    print("== Environment ==")
    print("python:", sys.version.split()[0])
    print("torch:", torch.__version__)
    print("cuda_available:", torch.cuda.is_available())
    print("torch_cuda_version:", torch.version.cuda)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. In Colab, use Runtime > Change runtime type > GPU."
        )

    device = torch.device("cuda")
    print("gpu:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
    return device


def _compile_cuda_extension() -> bool:
    cuda_dir = ROOT / "scao" / "cuda"
    try:
        _run([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=cuda_dir)
    except subprocess.CalledProcessError as exc:
        print("\nCUDA extension build failed; continuing with PyTorch fallback.")
        print(f"build_exit_code: {exc.returncode}")
        return False
    return True


def _check_fused_preconditioner(device: torch.device) -> None:
    print("\n== fused_kronecker_precond check ==")
    U = torch.eye(16, 4, device=device)
    s = torch.ones(4, device=device)
    G = torch.randn(16, 16, device=device)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        out = fused_kronecker_precond(U, s, U, s, G)

    fallback_warnings = [
        str(w.message) for w in caught
        if "compiled CUDA extension" in str(w.message)
    ]

    print("output_shape:", tuple(out.shape))
    print("identity_case:", torch.allclose(out, G, atol=1e-6))
    print("cuda_extension_loaded:", not fallback_warnings)
    if fallback_warnings:
        print("warning:", fallback_warnings[0])

    if not torch.isfinite(out).all():
        raise RuntimeError("fused_kronecker_precond produced non-finite values")


def _assert_model_finite(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if not torch.isfinite(param).all():
            raise RuntimeError(f"non-finite parameter detected: {name}")


def _run_mlp_smoke(device: torch.device, steps: int) -> None:
    print("\n== SCAO async MLP smoke ==")
    model = nn.Sequential(
        nn.Linear(1024, 2048),
        nn.GELU(),
        nn.Linear(2048, 1024),
    ).to(device)

    opt = SCAO(
        model.parameters(),
        lr=1e-3,
        warmup_steps=5,
        precond_freq=2,
        async_precond=True,
        noise_std_init=0.0,
        sparsity=0.0,
        lars_coeff=0.0,
        lookahead_k=0,
    )

    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    last_loss = math.nan

    for step in range(1, steps + 1):
        x = torch.randn(8, 1024, device=device)
        loss = model(x).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        last_loss = float(loss.detach())
        if not math.isfinite(last_loss):
            raise RuntimeError(f"non-finite MLP loss at step {step}: {last_loss}")

    opt.synchronize_precond()
    torch.cuda.synchronize()
    _assert_model_finite(model)

    print("steps:", steps)
    print("last_loss:", f"{last_loss:.6f}")
    print("time_s:", f"{time.time() - start:.2f}")
    print("peak_mb:", f"{torch.cuda.max_memory_allocated() / 1024**2:.1f}")


def _run_transformer_preset_smoke(device: torch.device, steps: int) -> None:
    print("\n== scao_1b Transformer smoke ==")
    model = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(
            d_model=768,
            nhead=12,
            dim_feedforward=3072,
            batch_first=True,
        ),
        num_layers=4,
    ).to(device)

    opt = scao_1b(
        model,
        lr=3e-4,
        max_precond_dim=1024,
        noise_std_init=0.0,
        sparsity=0.0,
        lars_coeff=0.0,
        lookahead_k=0,
    )

    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    last_loss = math.nan

    for step in range(1, steps + 1):
        x = torch.randn(2, 128, 768, device=device)
        loss = model(x).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        last_loss = float(loss.detach())
        if not math.isfinite(last_loss):
            raise RuntimeError(f"non-finite Transformer loss at step {step}: {last_loss}")

    opt.synchronize_precond()
    torch.cuda.synchronize()
    _assert_model_finite(model)

    print("steps:", steps)
    print("last_loss:", f"{last_loss:.6f}")
    print("time_s:", f"{time.time() - start:.2f}")
    print("peak_mb:", f"{torch.cuda.max_memory_allocated() / 1024**2:.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="SCAO Colab T4 smoke test")
    parser.add_argument("--compile-cuda-ext", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    device = _require_cuda()

    if args.compile_cuda_ext:
        _compile_cuda_extension()

    _check_fused_preconditioner(device)

    mlp_steps = 8 if args.quick else 20
    transformer_steps = 6 if args.quick else 30
    _run_mlp_smoke(device, mlp_steps)
    _run_transformer_preset_smoke(device, transformer_steps)

    print("\nSCAO Colab T4 smoke test: OK")


if __name__ == "__main__":
    main()
