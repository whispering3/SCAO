"""
SCAO Scale Presets
===================
Pre-tuned hyperparameter profiles tailored for model parameter scales
from sub-1B up to 125B+.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from .optimizer import SCAO


def _params_from_model_or_iterable(model_or_params: Any) -> Iterable:
    return cast(Iterable, (
        model_or_params.parameters()
        if hasattr(model_or_params, "parameters")
        else model_or_params
    ))


def _merge_preset_defaults(overrides: dict[str, Any], **defaults: Any) -> dict[str, Any]:
    merged = dict(defaults)
    merged.update(overrides)
    return merged


def scao_sub1b(model_or_params: Any, lr: float = 5e-4, **kw: Any) -> SCAO:
    """
    <1B params (125M–999M). RTX 3060/T4, CPU offload.
    Kronecker inversion is cheap at this scale — maximise update frequency.
    Lookahead disabled: these models converge fast enough that slow weights
    create net drag. k_min=4 handles narrow layers without falling back to
    diagonal; verify SparsePreconditioner has this guard before use.
    """
    from .optimizer import SCAO

    return SCAO(
        _params_from_model_or_iterable(model_or_params), lr=lr,
        **_merge_preset_defaults(
            kw,
            warmup_steps=20,
            min_precond_updates=3,
            precond_freq=5,
            k_min=4,
            k_max=64,
            max_precond_dim=1024,
            epsilon_sparse=0.10,
            sparsity=0.0,
            dynamic_sparsity=False,
            adaptive_warmup=False,
            lazy_precond=False,
            use_gsnr_clip=False,
            adaptive_rank=False,
            noise_std_init=0.0,
            lars_coeff=0.0,
            lookahead_k=0,
            beta3=0.98,
            tau=0.8,
        ),
    )


def scao_1b(model_or_params: Any, lr: float = 3e-4, **kw: Any) -> SCAO:
    """
    1B–3B params (Phi-2, Gemma-2B, TinyLlama). RTX 3090/A10, T4×2.
    """
    from .optimizer import SCAO

    return SCAO(
        _params_from_model_or_iterable(model_or_params), lr=lr,
        **_merge_preset_defaults(
            kw,
            warmup_steps=35,
            min_precond_updates=5,
            precond_freq=8,
            k_min=4,
            k_max=96,
            max_precond_dim=2048,
            epsilon_sparse=0.07,
            sparsity=0.0,
            dynamic_sparsity=False,
            adaptive_warmup=False,
            lazy_precond=False,
            use_gsnr_clip=False,
            adaptive_rank=False,
            noise_std_init=0.0,
            lars_coeff=0.0,
            lookahead_k=0,
            beta3=0.985,
            tau=0.9,
        ),
    )


def scao_3b(model_or_params: Any, lr: float = 2e-4, **kw: Any) -> SCAO:
    """3B params. Memory-bounded QLoRA profile for T4/A10 GPUs.

    LoRA matrices have one wide side and one rank-r side.  Limiting dense
    curvature blocks to 256 avoids retaining full hidden_size² EMAs for every
    adapter while preserving block-local preconditioning on the wide side.
    """
    from .optimizer import SCAO

    return SCAO(
        _params_from_model_or_iterable(model_or_params), lr=lr,
        **_merge_preset_defaults(
            kw,
            warmup_steps=20,
            blend_steps=30,
            precond_freq=10,
            min_precond_updates=2,
            max_precond_dim=256,
            use_int8_ema=True,
            factor_mode="wide",
            dynamic_sparsity=False,
            adaptive_warmup=False,
            lazy_precond=False,
            use_gsnr_clip=False,
            adaptive_rank=False,
            sparsity=0.0,
            noise_std_init=0.0,
            lars_coeff=0.0,
            lookahead_k=0,
        ),
    )


def scao_3b_low_memory(model_or_params: Any, lr: float = 2e-4, **kw: Any) -> SCAO:
    """3B QLoRA profile prioritising optimizer-state memory over curvature detail.

    This profile halves the wide curvature-block size relative to ``scao_3b``.
    Compare it against an AdamW baseline on the target workload before use.
    """
    from .optimizer import SCAO

    return SCAO(
        _params_from_model_or_iterable(model_or_params), lr=lr,
        **_merge_preset_defaults(
            kw,
            warmup_steps=20,
            blend_steps=30,
            precond_freq=20,
            min_precond_updates=2,
            max_precond_dim=128,
            use_int8_ema=True,
            factor_mode="wide",
            dynamic_sparsity=False,
            adaptive_warmup=False,
            lazy_precond=False,
            use_gsnr_clip=False,
            adaptive_rank=False,
            sparsity=0.0,
            noise_std_init=0.0,
            lars_coeff=0.0,
            lookahead_k=0,
        ),
    )


def scao_7b(model_or_params: Any, lr: float = 1e-4, **kw: Any) -> SCAO:
    """7B params. A100 40 GB, QLoRA or full fine-tune."""
    from .optimizer import SCAO

    return SCAO(
        _params_from_model_or_iterable(model_or_params), lr=lr,
        **_merge_preset_defaults(
            kw,
            warmup_steps=80,
            precond_freq=15,
            dynamic_sparsity=False,
            adaptive_warmup=False,
            lazy_precond=False,
            use_gsnr_clip=False,
            adaptive_rank=False,
            sparsity=0.0,
            noise_std_init=0.0,
            lars_coeff=0.0,
            lookahead_k=0,
        ),
    )


def scao_40b(model_or_params: Any, lr: float = 5e-5, **kw: Any) -> SCAO:
    """14B–70B params. 4–8×A100/H100."""
    from .optimizer import SCAO

    return SCAO(
        _params_from_model_or_iterable(model_or_params), lr=lr,
        **_merge_preset_defaults(
            kw,
            warmup_steps=100,
            blend_steps=50,
            precond_freq=20,
            k_min=4,
            k_max=64,
            max_precond_dim=2048,
            dynamic_sparsity=False,
            adaptive_warmup=False,
            lazy_precond=False,
            use_gsnr_clip=False,
            adaptive_rank=False,
            use_int8_ema=True,
            sparsity=0.0,
            noise_std_init=0.0,
            lars_coeff=0.0,
            lookahead_k=0,
        ),
    )


def scao_125b(model_or_params: Any, lr: float = 2e-5, **kw: Any) -> SCAO:
    """70B+ params. FSDP / Megatron, 32–64×H100."""
    from .optimizer import SCAO

    return SCAO(
        _params_from_model_or_iterable(model_or_params), lr=lr,
        **_merge_preset_defaults(
            kw,
            warmup_steps=200,
            precond_freq=50,
            k_min=4,
            k_max=32,
            max_precond_dim=1024,
            dynamic_sparsity=False,
            adaptive_warmup=False,
            lazy_precond=False,
            use_gsnr_clip=False,
            adaptive_rank=False,
            use_int8_ema=True,
            sparsity=0.0,
            noise_std_init=0.0,
            lars_coeff=0.0,
            lookahead_k=0,
            async_precond=True,
        ),
    )
