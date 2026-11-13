"""Small, typed configuration helpers for the common SCAO use cases."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class SCAOConfig:
    """A compact, serialisable configuration for :class:`scao.SCAO`.

    Use this for the usual training knobs instead of passing a long list of
    keyword arguments to ``SCAO``. Advanced options remain available on the
    optimizer constructor and can be supplied through :meth:`build`.
    """

    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    precond_freq: int = 20
    min_precond_updates: int = 2
    max_precond_dim: int = 2048
    k_min: int = 8
    k_max: int = 64
    tau: float | None = 1.0
    use_int8_ema: bool = False
    factor_mode: str = "full"
    async_precond: bool = True

    @classmethod
    def for_lora(cls, lr: float = 2e-4, **overrides: Any) -> SCAOConfig:
        """Return a memory-bounded starting point for LoRA/QLoRA adapters."""
        values: dict[str, Any] = {
            "lr": lr,
            "warmup_steps": 20,
            "precond_freq": 10,
            "min_precond_updates": 2,
            "max_precond_dim": 256,
            "k_min": 4,
            "k_max": 64,
            "use_int8_ema": True,
            "factor_mode": "wide",
        }
        values.update(overrides)
        return cls(**values)

    def to_kwargs(self) -> dict[str, Any]:
        """Return constructor-ready keyword arguments."""
        return asdict(self)

    def build(self, params: Any, **overrides: Any) -> Any:
        """Build an optimizer, optionally supplying advanced overrides."""
        from .optimizer import SCAO

        kwargs = self.to_kwargs()
        kwargs.update(overrides)
        return SCAO(params, **kwargs)
