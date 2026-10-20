"""Fair QLoRA benchmark: AdamW versus the memory-bounded SCAO 3B preset.

Run both optimizers with matching settings:
``python benchmark_4b.py --optimizer both --steps 300 --lr 2e-4 --seed 42``.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    DataCollatorForLanguageModeling, Trainer, TrainingArguments, set_seed,
)

import scao
from scao import scao_3b


def _optimizer_state_bytes(optimizer: torch.optim.Optimizer) -> int:
    """Count unique tensors retained by nested optimizer state objects."""
    visited: set[int] = set()

    def visit(value: Any) -> int:
        if id(value) in visited:
            return 0
        visited.add(id(value))
        if isinstance(value, torch.Tensor):
            return value.numel() * value.element_size()
        if isinstance(value, dict):
            return sum(visit(item) for item in value.values())
        if isinstance(value, (tuple, list)):
            return sum(visit(item) for item in value)
        if hasattr(value, "__dict__"):
            return visit(vars(value))
        return 0

    return visit(optimizer.state)


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _scao_source_fingerprint() -> tuple[str, str]:
    """Return an install-location-independent fingerprint of SCAO Python code."""
    package_dir = Path(scao.__file__).resolve().parent
    digest = hashlib.sha256()
    for source_file in sorted(package_dir.rglob("*.py")):
        relative_path = source_file.relative_to(package_dir).as_posix()
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source_file.read_bytes())
        digest.update(b"\0")
    return str(package_dir), digest.hexdigest()


def _write_result(
    name: str,
    args: argparse.Namespace,
    optimizer: torch.optim.Optimizer,
    train: Any,
    validation: Any,
    resolved_model_revision: str | None,
    train_metrics: dict[str, Any],
    eval_metrics: dict[str, Any],
    peak_gib: float,
    state_mib: float,
) -> Path:
    """Persist the inputs and measurements needed to reproduce one run."""
    output_dir = Path(args.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device_name = torch.cuda.get_device_name() if torch.cuda.is_available() else None
    source_path, source_fingerprint = _scao_source_fingerprint()
    payload = {
        "optimizer": name,
        "measurements": {
            "eval_loss": eval_metrics.get("eval_loss"),
            "peak_vram_gib": peak_gib,
            "optimizer_state_mib": state_mib,
            "train_runtime_seconds": train_metrics.get("train_runtime"),
            "train_steps_per_second": train_metrics.get("train_steps_per_second"),
        },
        "benchmark": {
            "model_id": args.model_id,
            "model_revision_requested": args.model_revision,
            "model_revision_resolved": resolved_model_revision,
            "dataset": "Salesforce/wikitext",
            "dataset_config": "wikitext-2-raw-v1",
            "train_split": args.train_split,
            "validation_split": args.validation_split,
            "train_fingerprint": getattr(train, "_fingerprint", None),
            "validation_fingerprint": getattr(validation, "_fingerprint", None),
            "steps": args.steps,
            "seed": args.seed,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "sequence_length": args.sequence_length,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_target_modules": args.lora_target_modules,
            "lr_warmup_steps": args.lr_warmup_steps,
            "lr_scheduler_type": str(args.lr_scheduler_type),
            "scao_max_precond_dim": args.scao_max_precond_dim,
            "scao_precond_freq": args.scao_precond_freq,
            "scao_use_int8_ema": args.scao_use_int8_ema,
            "scao_curvature_mode": "dense",
        },
        "optimizer_groups": [
            {
                key: value
                for key, value in group.items()
                if key != "params" and isinstance(value, (str, int, float, bool, type(None)))
            }
            for group in optimizer.param_groups
        ],
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": device_name,
            "transformers": _package_version("transformers"),
            "datasets": _package_version("datasets"),
            "peft": _package_version("peft"),
            "bitsandbytes": _package_version("bitsandbytes"),
            "scao": _package_version("scao"),
            "git_revision": _git_revision(),
            "scao_source_path": source_path,
            "scao_source_sha256": source_fingerprint,
            "source_label": args.source_label,
        },
    }
    run_name = name
    if name == "scao":
        run_name = (
            f"scao-{args.scao_factor_mode}-block{args.scao_max_precond_dim}"
            f"-freq{args.scao_precond_freq}"
            "-curvdense"
            f"-ema{'int8' if args.scao_use_int8_ema else 'fp32'}"
        )
    path = output_dir / f"{run_name}-seed{args.seed}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _build_model(
    model_id: str,
    revision: str | None,
    lora_rank: int,
    lora_alpha: int,
    lora_target_modules: tuple[str, ...],
) -> tuple[Any, Any, str | None]:
    source_kwargs = {"revision": revision} if revision else {}
    tokenizer = AutoTokenizer.from_pretrained(model_id, **source_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16),
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=quantization, device_map="auto", **source_kwargs
    )
    resolved_revision = getattr(model.config, "_commit_hash", None)
    if resolved_revision is None:
        resolved_revision = tokenizer.init_kwargs.get("_commit_hash")
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=lora_rank, lora_alpha=lora_alpha, target_modules=list(lora_target_modules),
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    ))
    return model, tokenizer, resolved_revision


def _dataset(tokenizer: Any, args: argparse.Namespace) -> tuple[Any, Any]:
    # Keep the transform closure limited to inputs that affect tokenization.
    # Capturing the whole ``args`` namespace makes Hugging Face derive a new
    # dataset fingerprint for unrelated optimizer toggles (for example, int8
    # versus FP32 EMA), even when every emitted token is identical.
    sequence_length = args.sequence_length

    def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(
            batch["text"], padding="max_length", truncation=True, max_length=sequence_length
        )

    def prepare(split: str) -> Any:
        dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
        dataset = dataset.filter(lambda row: bool(row["text"].strip()))
        return dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)

    return prepare(args.train_split), prepare(args.validation_split)


def _training_args(name: str, args: argparse.Namespace) -> TrainingArguments:
    """Build TrainingArguments across supported Transformers releases."""
    kwargs: dict[str, Any] = {
        "output_dir": str(Path(args.results_dir) / f"{name}-seed{args.seed}"),
        "overwrite_output_dir": True,
        "max_steps": args.steps,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.lr_warmup_steps,
        "lr_scheduler_type": args.lr_scheduler_type,
        "logging_steps": 10,
        "save_strategy": "no",
        "report_to": "none",
        "gradient_checkpointing": True,
        "seed": args.seed,
        "data_seed": args.seed,
    }
    signature = inspect.signature(TrainingArguments)
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "no"
    elif "evaluation_strategy" in signature.parameters:
        kwargs["evaluation_strategy"] = "no"
    return TrainingArguments(**{
        key: value for key, value in kwargs.items() if key in signature.parameters
    })


def _run(name: str, args: argparse.Namespace, train: Any, validation: Any) -> None:
    set_seed(args.seed)
    model, tokenizer, resolved_model_revision = _build_model(
        args.model_id,
        args.model_revision,
        args.lora_rank,
        args.lora_alpha,
        args.lora_target_modules,
    )
    trainable = [param for param in model.parameters() if param.requires_grad]
    if name == "adamw":
        optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            trainable, lr=args.lr, weight_decay=args.weight_decay
        )
        activation_step = None
    else:
        optimizer = scao_3b(
            trainable,
            lr=args.lr,
            weight_decay=args.weight_decay,
            max_precond_dim=args.scao_max_precond_dim,
            precond_freq=args.scao_precond_freq,
            factor_mode=args.scao_factor_mode,
            use_int8_ema=args.scao_use_int8_ema,
        )
        group = optimizer.param_groups[0]
        activation_step = max(
            group["warmup_steps"], group["precond_freq"] * group["min_precond_updates"]
        ) + 1

    training_args = _training_args(name, args)
    trainer = Trainer(
        model=model, args=training_args, train_dataset=train, eval_dataset=validation,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        optimizers=(optimizer, None),
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    train_metrics = trainer.train().metrics
    metrics = trainer.evaluate()
    peak_gib = torch.cuda.max_memory_allocated() / 1024**3
    state_mib = _optimizer_state_bytes(optimizer) / 1024**2
    print(
        f"{name:>5} | eval_loss={metrics['eval_loss']:.4f} | "
        f"peak_vram={peak_gib:.2f} GiB | optimizer_state={state_mib:.1f} MiB"
    )
    result_path = _write_result(
        name,
        args,
        optimizer,
        train,
        validation,
        resolved_model_revision,
        train_metrics,
        metrics,
        peak_gib,
        state_mib,
    )
    print(f"      reproducibility record: {result_path}")
    if activation_step is not None:
        print(f"      SCAO preconditioning begins no earlier than optimizer step {activation_step}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimizer", choices=("adamw", "scao", "both"), default="both")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-3B")
    parser.add_argument(
        "--model-revision", default="main",
        help="Hugging Face revision to load; the resolved commit is written to JSON.",
    )
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=2e-4, help="Identical LR for both optimizers.")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--train-split", default="train[:5%]")
    parser.add_argument("--validation-split", default="validation[:5%]")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj",
        help="Comma-separated projection modules for LoRA.",
    )
    parser.add_argument("--lr-warmup-steps", type=int, default=0)
    parser.add_argument("--lr-scheduler-type", default="linear")
    parser.add_argument(
        "--scao-max-precond-dim", type=int, default=256, choices=(64, 128, 256),
        help="SCAO curvature-block size; lower values reduce optimizer state.",
    )
    parser.add_argument(
        "--scao-precond-freq", type=int, default=10,
        help="SCAO curvature-update interval in optimizer steps.",
    )
    parser.add_argument(
        "--scao-factor-mode", choices=("full", "wide"), default="wide",
        help="Use full Kronecker factors or LoRA-oriented wide-side factors.",
    )
    parser.add_argument(
        "--scao-use-int8-ema",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Store curvature EMA in int8 (default). Use --no-scao-use-int8-ema "
            "to measure the FP32 speed-versus-memory trade-off."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--results-dir", default="scao_benchmark_4b_results",
        help="Directory for per-run reproducibility JSON records.",
    )
    parser.add_argument(
        "--source-label", default=None,
        help="Optional label for the source archive or checkout used in this run.",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This QLoRA benchmark requires a CUDA GPU.")
    if args.steps < 1:
        raise ValueError("--steps must be >= 1")
    if args.sequence_length < 1:
        raise ValueError("--sequence-length must be >= 1")
    if args.lora_rank < 1 or args.lora_alpha < 1:
        raise ValueError("--lora-rank and --lora-alpha must be >= 1")
    args.lora_target_modules = tuple(
        module.strip() for module in args.lora_target_modules.split(",") if module.strip()
    )
    if not args.lora_target_modules:
        raise ValueError("--lora-target-modules must contain at least one module")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=args.model_revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train, validation = _dataset(tokenizer, args)
    names = ("adamw", "scao") if args.optimizer == "both" else (args.optimizer,)
    print(
        f"Benchmark: model={args.model_id}@{args.model_revision}, steps={args.steps}, lr={args.lr:g}, "
        f"seed={args.seed}, scao_block={args.scao_max_precond_dim}, "
        f"scao_freq={args.scao_precond_freq}, scao_mode={args.scao_factor_mode}, "
        f"scao_int8_ema={args.scao_use_int8_ema}, scao_curvature_mode=dense"
    )
    for name in names:
        _run(name, args, train, validation)


if __name__ == "__main__":
    main()
