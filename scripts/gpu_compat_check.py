"""
SCAO GPU compatibility check.

Run this first on a new machine before fine-tuning:

    python scripts/gpu_compat_check.py

Optional:

    python scripts/gpu_compat_check.py --compile-cuda-ext
    python scripts/gpu_compat_check.py --strict-cuda-ext

The script reports the detected device, whether the optional CUDA extension is
loaded, whether SCAO can run through the PyTorch fallback, and a conservative
recommended config for the current GPU class.
"""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
import warnings
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], cwd: Path | None = None) -> bool:
    print(f"\n$ {' '.join(cmd)}")
    completed = subprocess.run(cmd, cwd=str(cwd or ROOT), check=False)
    return completed.returncode == 0


def _compile_cuda_extension() -> bool:
    cuda_dir = ROOT / "scao" / "cuda"
    if not cuda_dir.exists():
        print("cuda_extension_build: skipped, scao/cuda directory not found")
        return False
    return _run([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=cuda_dir)


def _device_profile() -> tuple[str, str]:
    if not torch.cuda.is_available():
        return "cpu", "CPU fallback"

    capability = torch.cuda.get_device_capability(0)
    arch = f"sm_{capability[0]}{capability[1]}"
    name = torch.cuda.get_device_name(0)
    return arch, name


def _recommended_config(arch: str) -> dict[str, int | bool]:
    base: dict[str, int | bool] = {
        "precond_freq": 20,
        "max_precond_dim": 1024,
        "k_min": 4,
        "k_max": 32,
        "async_precond": True,
    }
    if arch in {"sm_80", "sm_89", "sm_90"}:
        base["max_precond_dim"] = 2048
    if arch == "cpu":
        base["precond_freq"] = 50
        base["max_precond_dim"] = 512
        base["async_precond"] = False
    return base


def _support_status(arch: str) -> str:
    statuses = {
        "cpu": "supported via PyTorch fallback",
        "sm_75": "validated target: T4",
        "sm_80": "production target: A100",
        "sm_86": "production target: A10/A10G/RTX 3090",
        "sm_89": "production target: RTX 4090/L4/L40S",
        "sm_90": "production target: H100",
    }
    return statuses.get(arch, "not yet validated; fallback may work if PyTorch supports it")


def _check_cuda_extension_loaded() -> bool:
    try:
        importlib.import_module("scao.cuda._scao_cuda")
    except ImportError:
        return False
    return True


def _run_functional_check(strict_cuda_ext: bool) -> bool:
    from scao import SCAO
    from scao.cuda import fused_kronecker_precond

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ext_loaded = _check_cuda_extension_loaded()
    if strict_cuda_ext and not ext_loaded:
        print("functional_check: failed, CUDA extension was required but is not loaded")
        return False

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        U = torch.eye(16, 4, device=device)
        s = torch.ones(4, device=device)
        G = torch.randn(16, 16, device=device)
        out = fused_kronecker_precond(U, s, U, s, G)

    fallback = any("compiled CUDA extension" in str(item.message) for item in caught)
    identity_ok = torch.allclose(out, G, atol=1e-6)
    finite_ok = torch.isfinite(out).all().item()

    model = torch.nn.Linear(32, 16).to(device)
    optimizer = SCAO(
        model.parameters(),
        lr=1e-3,
        warmup_steps=2,
        precond_freq=2,
        k_max=8,
        noise_std_init=0.0,
        sparsity=0.0,
        lars_coeff=0.0,
        lookahead_k=0,
    )
    for _ in range(3):
        x = torch.randn(8, 32, device=device)
        loss = model(x).pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if hasattr(optimizer, "synchronize_precond"):
        optimizer.synchronize_precond()

    params_finite = all(torch.isfinite(param).all().item() for param in model.parameters())
    print("fused_precond_identity:", identity_ok)
    print("fused_precond_finite:", finite_ok)
    print("optimizer_step_finite:", params_finite)
    print("cuda_extension_loaded:", ext_loaded)
    print("using_pytorch_fallback:", fallback or not ext_loaded)
    return bool(identity_ok and finite_ok and params_finite)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check SCAO GPU compatibility")
    parser.add_argument("--compile-cuda-ext", action="store_true")
    parser.add_argument("--strict-cuda-ext", action="store_true")
    args = parser.parse_args()

    print("== Environment ==")
    print("python:", sys.version.split()[0])
    print("torch:", torch.__version__)
    print("cuda_available:", torch.cuda.is_available())
    print("torch_cuda_version:", torch.version.cuda)

    arch, device_name = _device_profile()
    print("device:", device_name)
    print("arch:", arch)
    print("support_status:", _support_status(arch))

    if args.compile_cuda_ext:
        built = _compile_cuda_extension()
        print("cuda_extension_build:", "ok" if built else "failed")

    print("\n== Functional check ==")
    ok = _run_functional_check(args.strict_cuda_ext)

    print("\n== Recommended SCAO config ==")
    for key, value in _recommended_config(arch).items():
        print(f"{key}: {value}")

    if ok:
        print("\nSCAO compatibility check: OK")
    else:
        raise SystemExit("SCAO compatibility check: FAILED")


if __name__ == "__main__":
    main()
