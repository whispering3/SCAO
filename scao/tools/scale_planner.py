"""
Memory planner for large-scale SCAO fine-tuning.

The planner estimates optimizer/preconditioner state for transformer-like
models before a run. It is meant for capacity planning on 40B-140B+ models,
where one accidental extra parameter copy can cost terabytes across a cluster.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class LayerShape:
    name: str
    rows: int
    cols: int
    count: int = 1

    @property
    def params(self) -> int:
        return self.rows * self.cols * self.count


@dataclass(frozen=True)
class ScaoConfig:
    k_max: int
    max_precond_dim: int
    use_int8_ema: bool
    lookahead_k: int


def _gb(num_bytes: float) -> float:
    return num_bytes / 1024**3


def _format_gb(num_bytes: float) -> str:
    return f"{_gb(num_bytes):,.2f} GB"


def _transformer_shapes(
    hidden_size: int,
    ffn_size: int,
    layers: int,
    vocab_size: int,
    tie_embeddings: bool,
) -> list[LayerShape]:
    shapes = [
        LayerShape("attn.qkv", 3 * hidden_size, hidden_size, layers),
        LayerShape("attn.out", hidden_size, hidden_size, layers),
        LayerShape("mlp.gate_up", 2 * ffn_size, hidden_size, layers),
        LayerShape("mlp.down", hidden_size, ffn_size, layers),
        LayerShape("token_embedding", vocab_size, hidden_size, 1),
    ]
    if not tie_embeddings:
        shapes.append(LayerShape("lm_head", vocab_size, hidden_size, 1))
    return shapes


def _infer_params_from_shapes(shapes: list[LayerShape]) -> int:
    dense_params = sum(shape.params for shape in shapes)
    return int(dense_params * 1.002)


def _preconditioner_bytes_for_matrix(rows: int, cols: int, config: ScaoConfig) -> int:
    if rows <= 1 or cols <= 1:
        return rows * cols * 4

    if max(rows, cols) > config.max_precond_dim:
        total = 0
        if rows >= cols:
            for offset in range(0, rows, config.max_precond_dim):
                block_rows = min(config.max_precond_dim, rows - offset)
                total += _preconditioner_bytes_for_matrix(block_rows, cols, config)
        else:
            for offset in range(0, cols, config.max_precond_dim):
                block_cols = min(config.max_precond_dim, cols - offset)
                total += _preconditioner_bytes_for_matrix(rows, block_cols, config)
        return total

    k = min(config.k_max, rows, cols)
    ema_bytes_per_elem = 1 if config.use_int8_ema else 4
    ema = (rows * rows + cols * cols) * ema_bytes_per_elem
    eigen = (rows * k + cols * k + 2 * k) * 4
    inverse_cache = 4 * k * 4
    scale_overhead = 16 if config.use_int8_ema else 0
    return int(ema + eigen + inverse_cache + scale_overhead)


def _preconditioner_bytes_for_shape(shape: LayerShape, config: ScaoConfig) -> int:
    return _preconditioner_bytes_for_matrix(shape.rows, shape.cols, config) * shape.count


def _estimate_preconditioner_bytes(
    shapes: list[LayerShape],
    config: ScaoConfig,
    total_params: int,
    precondition_embeddings: bool,
) -> tuple[int, str | None]:
    if shapes:
        total = 0
        for shape in shapes:
            is_embedding_like = "embedding" in shape.name or shape.name == "lm_head"
            if is_embedding_like and not precondition_embeddings:
                continue
            total += _preconditioner_bytes_for_shape(shape, config)
        return total, None

    bytes_per_param = 4.0 if config.use_int8_ema else 8.0
    warning = (
        "params-only estimate is conservative and shape-agnostic; pass "
        "--hidden-size --ffn-size --layers for capacity planning."
    )
    return int(total_params * bytes_per_param), warning


def _recommended_config(params_b: float) -> ScaoConfig:
    if params_b >= 100:
        return ScaoConfig(k_max=32, max_precond_dim=1024, use_int8_ema=True, lookahead_k=0)
    if params_b >= 40:
        return ScaoConfig(k_max=64, max_precond_dim=2048, use_int8_ema=True, lookahead_k=0)
    return ScaoConfig(k_max=96, max_precond_dim=2048, use_int8_ema=True, lookahead_k=0)


def _require_positive(name: str, value: int | float | None) -> None:
    if value is not None and value <= 0:
        raise SystemExit(f"{name} must be positive, got {value}.")


def _validate_args(args: argparse.Namespace) -> None:
    _require_positive("--params-b", args.params_b)
    _require_positive("--hidden-size", args.hidden_size)
    _require_positive("--ffn-size", args.ffn_size)
    _require_positive("--layers", args.layers)
    _require_positive("--vocab-size", args.vocab_size)
    _require_positive("--world-size", args.world_size)
    _require_positive("--gpu-memory-gb", args.gpu_memory_gb)
    _require_positive("--k-max", args.k_max)
    _require_positive("--max-precond-dim", args.max_precond_dim)
    if args.lookahead_k is not None and args.lookahead_k < 0:
        raise SystemExit(f"--lookahead-k must be non-negative, got {args.lookahead_k}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan SCAO memory for large fine-tuning")
    parser.add_argument("--params-b", type=float, default=None)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--ffn-size", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--untied-embeddings", action="store_true")
    parser.add_argument("--world-size", type=int, default=64)
    parser.add_argument("--gpu-memory-gb", type=float, default=80.0)
    parser.add_argument("--zero-stage", type=int, choices=(0, 1, 2, 3), default=3)
    parser.add_argument(
        "--precondition-embeddings",
        action="store_true",
        help="Include token embeddings/lm_head in SCAO preconditioner estimates.",
    )
    parser.add_argument("--k-max", type=int, default=None)
    parser.add_argument("--max-precond-dim", type=int, default=None)
    parser.add_argument("--fp32-ema", action="store_true")
    parser.add_argument("--lookahead-k", type=int, default=None)
    args = parser.parse_args()
    _validate_args(args)

    shape_args = (args.hidden_size, args.ffn_size, args.layers)
    has_shapes = all(value is not None for value in shape_args)
    if args.params_b is None and not has_shapes:
        raise SystemExit("Provide either --params-b or --hidden-size/--ffn-size/--layers.")
    if args.params_b is not None and has_shapes:
        print("warning: both --params-b and explicit shapes were provided; using explicit shapes.")

    shapes: list[LayerShape] = []
    if has_shapes:
        assert args.hidden_size is not None
        assert args.ffn_size is not None
        assert args.layers is not None
        shapes = _transformer_shapes(
            args.hidden_size,
            args.ffn_size,
            args.layers,
            args.vocab_size,
            tie_embeddings=not args.untied_embeddings,
        )
        total_params = _infer_params_from_shapes(shapes)
    else:
        assert args.params_b is not None
        total_params = int(args.params_b * 1_000_000_000)

    params_b = total_params / 1_000_000_000
    recommended = _recommended_config(params_b)
    config = ScaoConfig(
        k_max=args.k_max if args.k_max is not None else recommended.k_max,
        max_precond_dim=(
            args.max_precond_dim
            if args.max_precond_dim is not None
            else recommended.max_precond_dim
        ),
        use_int8_ema=not args.fp32_ema,
        lookahead_k=args.lookahead_k if args.lookahead_k is not None else recommended.lookahead_k,
    )

    param_bytes = total_params * 2
    grad_bytes = total_params * 2
    adam_moments = total_params * 8
    master_weights = total_params * 4
    lookahead_bytes = total_params * 2 if config.lookahead_k > 0 else 0
    precond_bytes, estimate_warning = _estimate_preconditioner_bytes(
        shapes,
        config,
        total_params,
        precondition_embeddings=args.precondition_embeddings,
    )

    replicated_total = param_bytes + grad_bytes + adam_moments + master_weights
    replicated_total += lookahead_bytes + precond_bytes

    shard_optimizer = args.zero_stage >= 1
    shard_grads = args.zero_stage >= 2
    shard_params = args.zero_stage >= 3

    per_gpu = 0.0
    per_gpu += param_bytes / args.world_size if shard_params else param_bytes
    per_gpu += grad_bytes / args.world_size if shard_grads else grad_bytes
    optimizer_state = adam_moments + master_weights + precond_bytes + lookahead_bytes
    per_gpu += optimizer_state / args.world_size if shard_optimizer else optimizer_state

    gpu_capacity = args.gpu_memory_gb * 1024**3
    headroom = gpu_capacity - per_gpu

    print("== SCAO large-scale planner ==")
    print("estimated_params:", f"{params_b:,.2f}B")
    print("world_size:", args.world_size)
    print("zero_stage:", args.zero_stage)
    print("gpu_memory:", f"{args.gpu_memory_gb:.1f} GB")
    print("")
    print("== Recommended SCAO knobs ==")
    print("k_max:", config.k_max)
    print("max_precond_dim:", config.max_precond_dim)
    print("use_int8_ema:", config.use_int8_ema)
    print("lookahead_k:", config.lookahead_k)
    print("precondition_embeddings:", args.precondition_embeddings)
    print("")
    print("== Cluster-level state estimate ==")
    print("params_bf16:", _format_gb(param_bytes))
    print("grads_bf16:", _format_gb(grad_bytes))
    print("adam_moments_fp32:", _format_gb(adam_moments))
    print("master_weights_fp32:", _format_gb(master_weights))
    print("scao_preconditioner:", _format_gb(precond_bytes))
    if estimate_warning:
        print("estimate_warning:", estimate_warning)
    print("lookahead_extra:", _format_gb(lookahead_bytes))
    print("total_no_sharding:", _format_gb(replicated_total))
    print("")
    print("== Per-GPU estimate ==")
    print("model_grad_optimizer:", _format_gb(per_gpu))
    print("activation_budget_remaining:", _format_gb(headroom))
    print("fits_before_activations:", headroom > 0)
    if headroom <= 0:
        min_world = math.ceil(per_gpu / gpu_capacity * args.world_size)
        print("suggested_min_world_size_before_activations:", min_world)
    elif headroom / gpu_capacity < 0.25:
        print("warning: less than 25% memory remains for activations, dataloader, and fragmentation")

    if config.lookahead_k > 0 and params_b >= 40:
        print("")
        print("warning: lookahead creates an extra parameter copy; disable it for 40B+ runs")


if __name__ == "__main__":
    main()
