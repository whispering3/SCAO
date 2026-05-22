"""
SCAO distributed training utilities.
=====================================
Helpers for ZeRO-3 (DeepSpeed) and FSDP (PyTorch ≥ 2.0) compatibility.

Usage with FSDP
---------------
    from scao.distributed import wrap_scao_for_fsdp

    model = FSDP(model, ...)
    optimizer = wrap_scao_for_fsdp(SCAO(model.parameters(), lr=1e-3))

Usage with DeepSpeed ZeRO-3
----------------------------
SCAO is compatible with ZeRO-3 out of the box as long as you do NOT enable
stage 3 optimizer state partitioning for preconditioner tensors (they must
live on the rank that owns the corresponding parameters).

In your DeepSpeed config, set:
    "zero_optimization": {
        "stage": 3,
        "stage3_param_persistence_threshold": 1e4
    }
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from .optimizer import SCAO


def _iter_preconditioners(optimizer: SCAO) -> Iterator[Any]:
    for state in optimizer.state.values():
        prec = state.get("preconditioner")
        if prec is not None:
            yield prec


def _iter_leaf_preconditioners(prec: Any) -> Iterator[Any]:
    if getattr(prec, "use_block_diagonal", False):
        for blk in prec._blocks:
            yield from _iter_leaf_preconditioners(blk)
    else:
        yield prec


def _preconditioner_step_sum(optimizer: SCAO) -> int:
    total = 0
    for prec in _iter_preconditioners(optimizer):
        for leaf in _iter_leaf_preconditioners(prec):
            total += int(getattr(leaf, "precond_step", 0))
    return total


def _sync_tensor_average(
    tensor: torch.Tensor,
    world_size: int,
    process_group: Any = None,
) -> None:
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=process_group)
    tensor.div_(world_size)


def _sync_leaf_preconditioner(
    prec: Any,
    world_size: int,
    process_group: Any = None,
) -> None:
    """
    Average curvature accumulators and recompute eigenfactors locally.

    Eigenvectors themselves are intentionally not averaged: their sign and
    ordering are not globally stable, so all-reducing them can destroy the
    orthonormal basis. The mathematically stable object to sync is the EMA
    curvature estimate.
    """
    from .utils import dequantize_sym_int8, quantize_sym_int8

    if getattr(prec, "use_kronecker", False):
        if getattr(prec, "use_int8_ema", False):
            L_ema = dequantize_sym_int8(prec.L_ema_q, prec.L_ema_scale)
            R_ema = dequantize_sym_int8(prec.R_ema_q, prec.R_ema_scale)
            _sync_tensor_average(L_ema, world_size, process_group)
            _sync_tensor_average(R_ema, world_size, process_group)
            prec.L_ema_q, prec.L_ema_scale = quantize_sym_int8(L_ema)
            prec.R_ema_q, prec.R_ema_scale = quantize_sym_int8(R_ema)
        else:
            _sync_tensor_average(prec.L_ema, world_size, process_group)
            _sync_tensor_average(prec.R_ema, world_size, process_group)

        if int(prec.precond_step) > 0:
            bias_factor = 1.0 - prec.rho ** int(prec.precond_step)
            prec._update_eigenfactors(bias_factor)
        return

    _sync_tensor_average(prec.diag_ema, world_size, process_group)


def sync_preconditioners(optimizer: SCAO, process_group: Any = None) -> None:
    """
    All-reduce preconditioner curvature EMAs across all ranks.

    In FSDP/ZeRO setups where parameters are sharded, each rank computes
    curvature from its local gradient shard.  This function averages those
    estimates so every rank has a globally consistent preconditioner.

    Call this AFTER optimizer.step() and BEFORE the next forward pass, or
    better: schedule it on the preconditioner update cadence (precond_freq).

    Supports both fp32 and int8 EMA accumulators (``use_int8_ema=True``).
    Int8 tensors are dequantized to float32 before the all-reduce, then
    re-quantized on each rank after averaging.

    Args:
        optimizer:  SCAO optimizer instance.
        process_group: optional distributed process group (default: global group).
    """
    if not dist.is_available() or not dist.is_initialized():
        return

    world_size = dist.get_world_size(group=process_group)
    if world_size == 1:
        return

    optimizer.synchronize_precond()

    for prec in _iter_preconditioners(optimizer):
        if getattr(prec, "use_block_diagonal", False):
            for blk in prec._blocks:
                _sync_leaf_preconditioner(blk, world_size, process_group)
            prec._k = sum(b._k for b in prec._blocks) // max(len(prec._blocks), 1)
        else:
            _sync_leaf_preconditioner(prec, world_size, process_group)


def wrap_scao_for_fsdp(optimizer: SCAO) -> SCAO:
    """
    Register a post-step hook that synchronises preconditioners after each
    curvature update in FSDP training.

    Returns the same optimizer (mutation in place), for chaining:
        optimizer = wrap_scao_for_fsdp(SCAO(...))
    """
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # noqa: F401
    except ImportError:
        warnings.warn(
            "torch.distributed.fsdp not available; FSDP wrapping skipped.",
            stacklevel=2,
        )
        return optimizer

    original_step = optimizer.step
    last_synced_precond_steps = _preconditioner_step_sum(optimizer)

    def patched_step(closure: Any = None) -> Any:
        nonlocal last_synced_precond_steps
        result = original_step(closure)
        optimizer.synchronize_precond()
        precond_steps = _preconditioner_step_sum(optimizer)
        if precond_steps != last_synced_precond_steps:
            sync_preconditioners(optimizer)
            last_synced_precond_steps = precond_steps
        return result

    optimizer.step = patched_step  # type: ignore[method-assign]
    return optimizer
