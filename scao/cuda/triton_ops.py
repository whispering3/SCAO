"""
SCAO Triton Kernels
===================
High-performance JIT-compiled kernels for SCAO operators using OpenAI Triton.
Allows GPU acceleration without requiring a precompiled C++/CUDA extension.
Falls back safely if Triton is not installed or CUDA is unavailable.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

_TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = torch.cuda.is_available()
except ImportError:
    triton = None  # type: ignore[assignment]
    tl = None      # type: ignore[assignment]


def is_triton_available() -> bool:
    """Return True if Triton is available and can execute on CUDA."""
    return _TRITON_AVAILABLE


if _TRITON_AVAILABLE:
    @triton.jit
    def _int8_ema_kernel(
        ema_q_ptr: Any,
        new_val_ptr: Any,
        out_q_ptr: Any,
        ema_scale: Any,
        rho: Any,
        n_elements: Any,
        inv_scale: Any,
        BLOCK_SIZE: tl.constexpr,
    ) -> None:

        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        # Load int8, convert to float32, dequantize
        q = tl.load(ema_q_ptr + offsets, mask=mask, other=0).to(tl.float32)
        dequant = q * ema_scale

        # Load new_val and blend
        nv = tl.load(new_val_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        updated = dequant * rho + nv

        # Quantize to int8
        q_out = tl.math.clamp(tl.math.round(updated * inv_scale), -127.0, 127.0).to(tl.int8)
        tl.store(out_q_ptr + offsets, q_out, mask=mask)


def int8_ema_update_triton(
    ema_q: Tensor,
    ema_scale: float,
    new_val: Tensor,
    rho: float,
) -> tuple[Tensor, float] | None:
    """
    Fused int8 EMA update using Triton when available.
    Returns (q_new, new_scale) or None if Triton execution fails.
    """
    if not _TRITON_AVAILABLE or not ema_q.is_cuda:
        return None

    try:
        # Pre-pass for scale: estimate updated magnitude
        # updated ≈ rho * (ema_q * scale) + new_val
        n_elements = ema_q.numel()
        fp32_approx = ema_q.to(torch.float32).mul_(ema_scale).mul_(rho).add_(new_val)
        abs_max = fp32_approx.abs().max().item()
        new_scale = abs_max / 127.0 if abs_max > 1e-30 else 1.0
        inv_scale = 1.0 / new_scale if new_scale > 1e-30 else 0.0

        out_q = torch.empty_like(ema_q)
        BLOCK_SIZE = 1024

        def grid(meta: dict[str, int]) -> tuple[int]:
            return ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)


        _int8_ema_kernel[grid](

            ema_q,
            new_val,
            out_q,
            ema_scale,
            rho,
            n_elements,
            inv_scale,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return out_q, new_scale
    except Exception:
        return None
