"""Create a paired Markdown summary from fair benchmark JSON records."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


_PAIR_FIELDS = (
    "model_id",
    "model_revision_requested",
    "dataset",
    "dataset_config",
    "train_split",
    "validation_split",
    "steps",
    "seed",
    "lr",
    "weight_decay",
    "batch_size",
    "gradient_accumulation_steps",
    "sequence_length",
    "lora_rank",
    "lora_alpha",
    "lora_target_modules",
)

_METADATA_FIELDS = (
    "model_revision_resolved",
    "train_fingerprint",
    "validation_fingerprint",
)


def _pair_key(record: dict[str, Any]) -> tuple[str, ...]:
    """Group runs by their declared, optimizer-invariant configuration.

    Dataset fingerprints are deliberately not included here. Hugging Face
    fingerprints can differ between separate Python processes despite loading
    the same declared dataset and tokenization settings, which is common when
    an SCAO ablation is run after a paired AdamW/SCAO baseline. They are
    compared separately and surfaced in the status column.
    """
    benchmark = record["benchmark"]
    environment = record["environment"]
    return tuple(
        json.dumps(benchmark.get(field), sort_keys=True)
        for field in _PAIR_FIELDS
    ) + (str(environment.get("gpu")),)


def _metadata_mismatches(adam: dict[str, Any], scao: dict[str, Any]) -> list[str]:
    """Return reproducibility metadata differences that need human review."""
    adam_benchmark = adam["benchmark"]
    scao_benchmark = scao["benchmark"]
    return [
        field
        for field in _METADATA_FIELDS
        if adam_benchmark.get(field) != scao_benchmark.get(field)
    ]


def _format(value: float | None, places: int = 4) -> str:
    return "—" if value is None else f"{value:.{places}f}"


def _records(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if {"optimizer", "benchmark", "environment", "measurements"} <= record.keys():
                record["_path"] = str(path)
                records.append(record)
        except (OSError, json.JSONDecodeError):
            continue
    return records


def render(records: list[dict[str, Any]]) -> str:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[_pair_key(record)].append(record)

    lines = ["# Paired SCAO Benchmark Summary", ""]
    lines.append("| GPU | Model | Seed | Configuration | AdamW loss | SCAO loss | Δ loss | SCAO state | SCAO time | Status |")
    lines.append("| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for key, group in sorted(groups.items()):
        adam = next((record for record in group if record["optimizer"] == "adamw"), None)
        scao_records = [record for record in group if record["optimizer"] == "scao"]
        for scao in scao_records:
            benchmark = scao["benchmark"]
            measurements = scao["measurements"]
            adam_loss = adam["measurements"].get("eval_loss") if adam else None
            scao_loss = measurements.get("eval_loss")
            delta = scao_loss - adam_loss if adam_loss is not None and scao_loss is not None else None
            curvature_mode = benchmark.get("scao_curvature_mode", "dense")
            ema_storage = (
                "n/a"
                if curvature_mode == "low_rank"
                else ("int8" if benchmark.get("scao_use_int8_ema", True) else "fp32")
            )
            config = (
                f"{scao['optimizer_groups'][0].get('factor_mode', 'unknown')} / "
                f"block {benchmark.get('scao_max_precond_dim')} / "
                f"freq {benchmark.get('scao_precond_freq')} / "
                f"curvature {curvature_mode} / EMA {ema_storage}"
            )
            if adam is None:
                status = "unpaired: AdamW record missing"
            else:
                mismatches = _metadata_mismatches(adam, scao)
                status = (
                    "paired"
                    if not mismatches
                    else f"paired: metadata differs ({', '.join(mismatches)})"
                )
            lines.append(
                "| {gpu} | {model} | {seed} | {config} | {adam_loss} | {scao_loss} | "
                "{delta} | {state} MiB | {runtime} s | {status} |".format(
                    gpu=scao["environment"].get("gpu", "unknown"),
                    model=benchmark.get("model_id", "unknown"),
                    seed=benchmark.get("seed", "—"),
                    config=config,
                    adam_loss=_format(adam_loss),
                    scao_loss=_format(scao_loss),
                    delta=_format(delta),
                    state=_format(measurements.get("optimizer_state_mib"), 1),
                    runtime=_format(measurements.get("train_runtime_seconds"), 1),
                    status=status,
                )
            )
    lines.extend(["", f"Records read: {len(records)}", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path, help="Directory containing benchmark JSON files.")
    parser.add_argument("--output", type=Path, required=True, help="Markdown report path.")
    args = parser.parse_args()
    records = _records(sorted(args.results_dir.glob("*.json")))
    if not records:
        raise SystemExit(f"No valid benchmark JSON records found in {args.results_dir}")
    args.output.write_text(render(records), encoding="utf-8")
    print(f"Wrote {args.output} from {len(records)} record(s).")


if __name__ == "__main__":
    main()
