"""Environment and CUDA readiness checks exposed as ``scao-doctor``."""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys

import torch


def _extension_loaded() -> bool:
    try:
        importlib.import_module("scao.cuda._scao_cuda")
    except ImportError:
        return False
    return True


def _build_extension() -> bool:
    completed = subprocess.run(
        [sys.executable, "-m", "scao.cuda.build"], check=False
    )
    return completed.returncode == 0


def _smoke_test() -> bool:
    from scao import SCAO

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.nn.Linear(16, 8).to(device)
    optimizer = SCAO(
        model.parameters(), lr=1e-3, warmup_steps=1, precond_freq=1,
        max_precond_dim=32, k_max=4,
    )
    for _ in range(2):
        optimizer.zero_grad()
        loss = model(torch.randn(4, 16, device=device)).square().mean()
        loss.backward()
        optimizer.step()
    optimizer.synchronize_precond()
    return all(torch.isfinite(param).all().item() for param in model.parameters())


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the SCAO runtime and optional CUDA path")
    parser.add_argument("--build-cuda", action="store_true", help="Build the optional CUDA extension")
    parser.add_argument("--strict-cuda", action="store_true", help="Fail when CUDA extension is unavailable")
    args = parser.parse_args()

    cuda_available = torch.cuda.is_available()
    print("python:", sys.version.split()[0])
    print("torch:", torch.__version__)
    print("cuda_available:", cuda_available)
    print("torch_cuda_version:", torch.version.cuda)
    if cuda_available:
        print("device:", torch.cuda.get_device_name(0))
        major, minor = torch.cuda.get_device_capability(0)
        print("compute_capability:", f"sm_{major}{minor}")

    if args.build_cuda:
        print("cuda_extension_build:", "ok" if _build_extension() else "failed")

    extension_loaded = _extension_loaded()
    print("cuda_extension_loaded:", extension_loaded)
    print("pytorch_fallback_available:", True)
    if args.strict_cuda and not extension_loaded:
        raise SystemExit("CUDA extension is required but was not loaded.")

    ok = _smoke_test()
    print("optimizer_smoke_test:", "ok" if ok else "failed")
    if not ok:
        raise SystemExit("SCAO environment check failed.")


if __name__ == "__main__":
    main()
