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
            sparsity=0.4,
            dynamic_sparsity=True,
            adaptive_warmup=True,
            warmup_stability_threshold=0.08,
            warmup_patience=3,
            lazy_precond=False,
            use_gsnr_clip=False,
            adaptive_rank=False,
            noise_std_init=0.005,
            noise_anneal=0.995,
            lars_coeff=5e-4,
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
            sparsity=0.50,
            dynamic_sparsity=True,
            adaptive_warmup=True,
            warmup_stability_threshold=0.06,
            warmup_patience=4,
            lazy_precond=False,
            use_gsnr_clip=False,
            adaptive_rank=False,
            noise_std_init=0.008,
            noise_anneal=0.997,
            lars_coeff=8e-4,
            lookahead_k=3,
            lookahead_alpha=0.4,
            beta3=0.985,
            tau=0.9,
        ),
    )


def scao_3b(model_or_params: Any, lr: float = 2e-4, **kw: Any) -> SCAO:
    """3B params. T4/A10 16–24 GB, QLoRA."""
    from .optimizer import SCAO

    return SCAO(
        _params_from_model_or_iterable(model_or_params), lr=lr,
        **_merge_preset_defaults(
            kw,
            warmup_steps=50,
            blend_steps=30,
            precond_freq=10,
            dynamic_sparsity=True,
            adaptive_warmup=True,
            warmup_patience=3,
            lazy_precond=False,
            use_gsnr_clip=False,
            adaptive_rank=False,
            sparsity=0.6,
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
            dynamic_sparsity=True,
            adaptive_warmup=True,
            lazy_precond=False,
            use_gsnr_clip=True,
            gsnr_threshold=0.4,
            adaptive_rank=True,
            sparsity=0.65,
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
            dynamic_sparsity=True,
            adaptive_warmup=True,
            lazy_precond=True,
            lazy_delta_threshold=0.08,
            lazy_max_skip=40,
            use_gsnr_clip=True,
            gsnr_threshold=0.5,
            adaptive_rank=True,
            use_int8_ema=True,
            sparsity=0.75,
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
            dynamic_sparsity=True,
            adaptive_warmup=True,
            lazy_precond=True,
            lazy_delta_threshold=0.05,
            lazy_max_skip=60,
            use_gsnr_clip=True,
            gsnr_threshold=0.6,
            adaptive_rank=True,
            use_int8_ema=True,
            sparsity=0.80,
            lookahead_k=0,
            async_precond=True,
        ),
    )
