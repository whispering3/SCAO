"""
Microbenchmark for the SCAO preconditioner hot path.

Measures repeated precondition() calls after eigenfactors have already been
computed, which is the per-step overhead users pay during training.
"""

from __future__ import annotations

import argparse
import statistics
import time
import warnings

import torch

from scao.preconditioner import SparsePreconditioner


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _build_preconditioner(
    dim: int,
    rank: int,
    device: torch.device,
) -> tuple[SparsePreconditioner, torch.Tensor]:
    param = torch.empty(dim, dim, device=device)
    grad = torch.randn_like(param)
    precond = SparsePreconditioner(
        param,
        k_min=min(rank, dim),
        k_max=min(rank, dim),
        max_precond_dim=dim,
        rho=0.99,
        use_int8_ema=False,
    )
    precond.update_curvature(grad)
    return precond, grad


def _bench_precondition(
    precond: SparsePreconditioner,
    grad: torch.Tensor,
    warmup: int,
    iters: int,
) -> list[float]:
    device = grad.device
    fallback_warned = False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        for _ in range(warmup):
            out = precond.precondition(grad)
            if not torch.isfinite(out).all():
                raise RuntimeError("precondition() produced non-finite values during warmup")
        _synchronize(device)
        fallback_warned = any("compiled CUDA extension" in str(item.message) for item in caught)

    times_ms: list[float] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        for _ in range(iters):
            start = time.perf_counter()
            out = precond.precondition(grad)
            _synchronize(device)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if not torch.isfinite(out).all():
                raise RuntimeError("precondition() produced non-finite values")
            times_ms.append(elapsed_ms)
        fallback_warned = fallback_warned or any(
            "compiled CUDA extension" in str(item.message) for item in caught
        )
    print("using_pytorch_fallback:", fallback_warned)
    return times_ms


def main() -> None:
    parser = argparse.ArgumentParser(description="SCAO preconditioner microbenchmark")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    if args.dim < 2:
        raise SystemExit(f"--dim must be at least 2, got {args.dim}.")
    if args.rank < 1:
        raise SystemExit(f"--rank must be positive, got {args.rank}.")
    if args.rank > args.dim:
        raise SystemExit(f"--rank must be <= --dim, got rank={args.rank}, dim={args.dim}.")
    if args.warmup < 0:
        raise SystemExit(f"--warmup must be non-negative, got {args.warmup}.")
    if args.iters < 1:
        raise SystemExit(f"--iters must be positive, got {args.iters}.")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)
        torch.cuda.reset_peak_memory_stats(device)

    print("== SCAO preconditioner microbenchmark ==")
    print("torch:", torch.__version__)
    print("device:", device)
    if device.type == "cuda":
        print("gpu:", torch.cuda.get_device_name(device))
    print("dim:", args.dim)
    print("rank:", args.rank)

    precond, grad = _build_preconditioner(args.dim, args.rank, device)
    times_ms = _bench_precondition(precond, grad, args.warmup, args.iters)

    print("iters:", args.iters)
    print("mean_ms:", f"{statistics.mean(times_ms):.4f}")
    print("median_ms:", f"{statistics.median(times_ms):.4f}")
    print("min_ms:", f"{min(times_ms):.4f}")
    print("max_ms:", f"{max(times_ms):.4f}")
    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2
        print("peak_mb:", f"{peak_mb:.1f}")


if __name__ == "__main__":
    main()
