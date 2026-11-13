"""
Microbenchmark for the SCAO preconditioner hot path.

Measures repeated precondition() calls after eigenfactors have already been
computed, which is the per-step overhead users pay during training.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import warnings
from pathlib import Path

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


def _bench_curvature(
    precond: SparsePreconditioner,
    grad: torch.Tensor,
    warmup: int,
    iters: int,
) -> list[float]:
    """Measure the scheduled curvature-update cost, including eigendecomposition."""
    device = grad.device
    for _ in range(warmup):
        precond.update_curvature(grad)
    _synchronize(device)

    times_ms: list[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        precond.update_curvature(grad)
        _synchronize(device)
        times_ms.append((time.perf_counter() - start) * 1000.0)
    return times_ms


def _bench_optimizer_step(
    dim: int, rank: int, device: torch.device, warmup: int, iters: int
) -> list[float]:
    """Measure the end-to-end optimizer path users pay during training."""
    from scao import SCAO

    parameter = torch.nn.Parameter(torch.empty(dim, dim, device=device).normal_())
    optimizer = SCAO(
        [parameter], lr=1e-3, warmup_steps=0, min_precond_updates=0,
        precond_freq=1, k_min=min(rank, dim), k_max=min(rank, dim),
        max_precond_dim=dim, async_precond=False,
    )
    for _ in range(warmup):
        parameter.grad = torch.randn_like(parameter)
        optimizer.step()
    _synchronize(device)

    times_ms: list[float] = []
    for _ in range(iters):
        parameter.grad = torch.randn_like(parameter)
        start = time.perf_counter()
        optimizer.step()
        _synchronize(device)
        times_ms.append((time.perf_counter() - start) * 1000.0)
    return times_ms


def _print_summary(label: str, times_ms: list[float]) -> dict[str, float]:
    summary = {
        "mean_ms": statistics.mean(times_ms),
        "median_ms": statistics.median(times_ms),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
    }
    for metric, value in summary.items():
        print(f"{label}_{metric}:", f"{value:.4f}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="SCAO preconditioner microbenchmark")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--operation", choices=("precondition", "curvature", "step", "all"), default="all",
        help="Operation to measure; 'all' reports each cost independently.",
    )
    parser.add_argument(
        "--json", type=Path, default=None,
        help="Write machine-readable benchmark results to this path.",
    )
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

    print("iters:", args.iters)
    results: dict[str, object] = {
        "torch": torch.__version__,
        "device": str(device),
        "dim": args.dim,
        "rank": args.rank,
        "warmup": args.warmup,
        "iters": args.iters,
        "operation": args.operation,
        "measurements_ms": {},
    }
    measurements = results["measurements_ms"]
    assert isinstance(measurements, dict)
    if args.operation in {"precondition", "curvature", "all"}:
        precond, grad = _build_preconditioner(args.dim, args.rank, device)
        if args.operation in {"precondition", "all"}:
            measurements["precondition"] = _print_summary(
                "precondition", _bench_precondition(precond, grad, args.warmup, args.iters)
            )
        if args.operation in {"curvature", "all"}:
            measurements["curvature"] = _print_summary(
                "curvature", _bench_curvature(precond, grad, args.warmup, args.iters)
            )
    if args.operation in {"step", "all"}:
        measurements["optimizer_step"] = _print_summary(
            "optimizer_step",
            _bench_optimizer_step(args.dim, args.rank, device, args.warmup, args.iters),
        )
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("json:", args.json)
    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2
        print("peak_mb:", f"{peak_mb:.1f}")


if __name__ == "__main__":
    main()
