from __future__ import annotations

from scao.tools.scale_planner import (
    LayerShape,
    ScaoConfig,
    _estimate_preconditioner_bytes,
    _preconditioner_bytes_for_matrix,
)


def test_params_only_planner_estimate_is_conservative_and_warns():
    config = ScaoConfig(
        k_max=32,
        max_precond_dim=1024,
        use_int8_ema=True,
        lookahead_k=0,
    )

    estimate, warning = _estimate_preconditioner_bytes(
        shapes=[],
        config=config,
        total_params=140_000_000_000,
        precondition_embeddings=False,
    )

    assert estimate == 140_000_000_000 * 4
    assert warning is not None
    assert "shape-agnostic" in warning


def test_shape_based_planner_can_skip_embeddings():
    config = ScaoConfig(
        k_max=32,
        max_precond_dim=1024,
        use_int8_ema=True,
        lookahead_k=0,
    )
    shapes = [
        LayerShape("mlp.down", 1024, 4096, 2),
        LayerShape("token_embedding", 32000, 1024, 1),
        LayerShape("lm_head", 32000, 1024, 1),
    ]

    without_embeddings, warning = _estimate_preconditioner_bytes(
        shapes=shapes,
        config=config,
        total_params=sum(shape.params for shape in shapes),
        precondition_embeddings=False,
    )
    with_embeddings, _ = _estimate_preconditioner_bytes(
        shapes=shapes,
        config=config,
        total_params=sum(shape.params for shape in shapes),
        precondition_embeddings=True,
    )

    expected_without = _preconditioner_bytes_for_matrix(1024, 4096, config) * 2
    assert without_embeddings == expected_without
    assert with_embeddings > without_embeddings
    assert warning is None
