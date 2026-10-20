"""
SCAO - Sparse Curvature-Aware Adaptive Optimizer
=================================================
PyTorch optimizer with Kronecker-factored curvature preconditioning.
Benchmark against an AdamW baseline for each workload.

Quickstart
----------
    import scao

    optimizer = scao.SCAO(model.parameters(), lr=3e-4)

    # or, explicit:
    from scao import SCAO
    optimizer = SCAO(
        model.parameters(),
        lr=3e-4,
        weight_decay=0.1,
        warmup_steps=100,
        precond_freq=10,
    )

Paper
-----
SCAO: Sparse Curvature-Aware Adaptive Optimization for Large-Scale Models
Zenodo 2026
"""

from . import logging as scao_logging
from .optimizer import (
    SCAO,
    scao_1b,
    scao_3b,
    scao_7b,
    scao_40b,
    scao_125b,
    scao_sub1b,
)
from .preconditioner import SparsePreconditioner
from .presets import scao_3b_low_memory
from .utils import adaptive_rank, matrix_power_neg_quarter

__version__ = "0.2.4"
__author__ = "Danilo Souza"
__license__ = "Apache-2.0"

__all__ = [
    # Main API — this is all most users need
    "SCAO",
    "scao_sub1b", "scao_1b", "scao_3b", "scao_3b_low_memory", "scao_7b", "scao_40b", "scao_125b",
    # Advanced / internals
    "SparsePreconditioner",
    "matrix_power_neg_quarter",
    "adaptive_rank",
    "scao_logging",
    # Metadata
    "__version__",
]

