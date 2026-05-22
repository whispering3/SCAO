"""
Real-user Colab test for SCAO on a T4 GPU.

This script runs a small Hugging Face fine-tuning job using SCAO exactly like
an end user would: model + dataset + Trainer + generation after training.

Colab usage from the repository root:

    !pip install -U "transformers>=4.30" "datasets>=2.0" accelerate -q
    !python scripts/colab_real_user_test.py

Optional:

    !python scripts/colab_real_user_test.py --quick
    !python scripts/colab_real_user_test.py --max-steps 200 --batch-size 2
    !python scripts/colab_real_user_test.py --compare-adamw
"""

from __future__ import annotations

import argparse
import inspect
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch


def _require_cuda() -> torch.device:
    print("== Environment ==")
    print("python:", sys.version.split()[0])
    print("torch:", torch.__version__)
    print("cuda_available:", torch.cuda.is_available())
    print("torch_cuda_version:", torch.version.cuda)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. In Colab, use Runtime > Change runtime type > GPU."
        )

    device = torch.device("cuda")
    print("gpu:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
    return device


def _load_dataset(tokenizer: Any, block_size: int, dataset_rows: int) -> Any:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "datasets is required. Install it with: "
            'pip install -U "datasets>=2.0"'
        ) from exc

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    dataset = dataset.filter(lambda row: bool(row["text"].strip()))
    if dataset_rows > 0:
        dataset = dataset.select(range(min(dataset_rows, len(dataset))))

    def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=block_size,
            padding="max_length",
        )

    tokenized = dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing WikiText-2",
    )
    return tokenized.train_test_split(test_size=0.05, seed=42)


def _build_model_and_tokenizer(model_name: str) -> tuple[Any, Any]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "transformers is required. Install it with: "
            'pip install -U "transformers>=4.30" accelerate'
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def _make_training_args(args: argparse.Namespace, output_dir: str) -> Any:
    from transformers import TrainingArguments

    kwargs: dict[str, Any] = {
        "output_dir": output_dir,
        "overwrite_output_dir": True,
        "max_steps": args.max_steps,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.lr,
        "warmup_steps": args.warmup_steps,
        "weight_decay": args.weight_decay,
        "fp16": True,
        "logging_steps": max(1, args.log_every),
        "eval_steps": max(1, args.max_steps // 2),
        "save_strategy": "no",
        "report_to": "none",
        "remove_unused_columns": False,
        "dataloader_num_workers": 2,
        "seed": args.seed,
    }

    signature = inspect.signature(TrainingArguments)
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "steps"
    else:
        kwargs["evaluation_strategy"] = "steps"

    return TrainingArguments(**kwargs)


def _train_with_scao(
    model: Any,
    tokenizer: Any,
    data: Any,
    args: argparse.Namespace,
    output_dir: str,
) -> dict[str, float]:
    from transformers import DataCollatorForLanguageModeling, Trainer

    import scao
    from scao.integrations.huggingface import get_scao_optimizer

    print("\n== SCAO package ==")
    print("version:", getattr(scao, "__version__", "unknown"))
    print("path:", getattr(scao, "__file__", "unknown"))

    training_args = _make_training_args(args, output_dir)
    optimizer, scheduler = get_scao_optimizer(
        model,
        training_args,
        scao_kwargs={
            "precond_freq": args.precond_freq,
            "min_precond_updates": 2,
            "max_precond_dim": args.max_precond_dim,
            "k_min": args.k_min,
            "k_max": args.k_max,
            "noise_std_init": 0.0,
            "sparsity": 0.0,
            "lars_coeff": 0.0,
            "lookahead_k": 0,
            "async_precond": True,
        },
        num_training_steps=args.max_steps,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data["train"],
        eval_dataset=data["test"],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        optimizers=(optimizer, scheduler),
    )

    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    train_result = trainer.train()
    if hasattr(optimizer, "synchronize_precond"):
        optimizer.synchronize_precond()
    torch.cuda.synchronize()

    eval_metrics = trainer.evaluate()
    train_loss = float(train_result.training_loss)
    eval_loss = float(eval_metrics["eval_loss"])
    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    elapsed_s = time.time() - start

    if not math.isfinite(train_loss) or not math.isfinite(eval_loss):
        raise RuntimeError(f"non-finite loss: train={train_loss}, eval={eval_loss}")

    return {
        "train_loss": train_loss,
        "eval_loss": eval_loss,
        "peak_mb": peak_mb,
        "elapsed_s": elapsed_s,
    }


def _train_with_adamw(
    model: Any,
    tokenizer: Any,
    data: Any,
    args: argparse.Namespace,
    output_dir: str,
) -> dict[str, float]:
    from transformers import DataCollatorForLanguageModeling, Trainer

    training_args = _make_training_args(args, output_dir)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data["train"],
        eval_dataset=data["test"],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    train_result = trainer.train()
    torch.cuda.synchronize()
    eval_metrics = trainer.evaluate()

    return {
        "train_loss": float(train_result.training_loss),
        "eval_loss": float(eval_metrics["eval_loss"]),
        "peak_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "elapsed_s": time.time() - start,
    }


def _generate_sample(model: Any, tokenizer: Any, prompt: str) -> None:
    print("\n== Generation sample ==")
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
        )
    print(tokenizer.decode(output[0], skip_special_tokens=True))


def _print_metrics(name: str, metrics: dict[str, float]) -> None:
    print(f"\n== {name} result ==")
    print("train_loss:", f"{metrics['train_loss']:.4f}")
    print("eval_loss:", f"{metrics['eval_loss']:.4f}")
    print("time_s:", f"{metrics['elapsed_s']:.1f}")
    print("peak_mb:", f"{metrics['peak_mb']:.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-user SCAO Colab T4 test")
    parser.add_argument("--model", default="distilgpt2")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--dataset-rows", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--precond-freq", type=int, default=10)
    parser.add_argument("--max-precond-dim", type=int, default=1024)
    parser.add_argument("--k-min", type=int, default=4)
    parser.add_argument("--k-max", type=int, default=32)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compare-adamw", action="store_true")
    parser.add_argument(
        "--prompt",
        default="Sparse curvature-aware optimization helps language models",
    )
    args = parser.parse_args()

    if args.quick:
        args.max_steps = min(args.max_steps, 20)
        args.dataset_rows = min(args.dataset_rows, 300)
        args.log_every = min(args.log_every, 5)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    _require_cuda()
    torch.manual_seed(args.seed)

    print("\n== Loading model and dataset ==")
    model, tokenizer = _build_model_and_tokenizer(args.model)
    model.to("cuda")
    data = _load_dataset(tokenizer, args.block_size, args.dataset_rows)

    with tempfile.TemporaryDirectory(prefix="scao-real-user-") as tmp:
        scao_metrics = _train_with_scao(
            model,
            tokenizer,
            data,
            args,
            str(Path(tmp) / "scao"),
        )
        _print_metrics("SCAO", scao_metrics)
        _generate_sample(model, tokenizer, args.prompt)

        if args.compare_adamw:
            print("\n== AdamW comparison ==")
            adamw_model, adamw_tokenizer = _build_model_and_tokenizer(args.model)
            adamw_model.to("cuda")
            adamw_metrics = _train_with_adamw(
                adamw_model,
                adamw_tokenizer,
                data,
                args,
                str(Path(tmp) / "adamw"),
            )
            _print_metrics("AdamW", adamw_metrics)

    print("\nSCAO real-user Colab test: OK")


if __name__ == "__main__":
    main()
