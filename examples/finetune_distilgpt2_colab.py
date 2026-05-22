"""
Lightweight SCAO fine-tuning example for Google Colab T4.

Install dependencies:

    pip install -U scao "transformers>=4.30" "datasets>=2.0" accelerate

Run:

    python examples/finetune_distilgpt2_colab.py

This is intentionally small: it validates the real user path with Hugging Face
Trainer, public data, fp16, and conservative SCAO defaults.
"""

from __future__ import annotations

import argparse
import inspect
import os
from typing import Any

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from scao.integrations.huggingface import get_scao_optimizer


def _training_args(args: argparse.Namespace) -> TrainingArguments:
    kwargs: dict[str, Any] = {
        "output_dir": args.output_dir,
        "overwrite_output_dir": True,
        "max_steps": args.max_steps,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.lr,
        "warmup_steps": args.warmup_steps,
        "weight_decay": args.weight_decay,
        "fp16": torch.cuda.is_available(),
        "logging_steps": args.logging_steps,
        "eval_steps": max(1, args.max_steps // 2),
        "save_strategy": "no",
        "report_to": "none",
        "remove_unused_columns": False,
        "seed": args.seed,
    }

    signature = inspect.signature(TrainingArguments)
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "steps"
    elif "evaluation_strategy" in signature.parameters:
        kwargs["evaluation_strategy"] = "steps"

    kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return TrainingArguments(**kwargs)


def _dataset(tokenizer: Any, rows: int, block_size: int) -> Any:
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    dataset = dataset.filter(lambda row: bool(row["text"].strip()))
    dataset = dataset.select(range(min(rows, len(dataset))))

    def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=block_size,
            padding="max_length",
        )

    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
    return tokenized.train_test_split(test_size=0.05, seed=42)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune distilgpt2 with SCAO")
    parser.add_argument("--model", default="distilgpt2")
    parser.add_argument("--output-dir", default="/content/scao-distilgpt2")
    parser.add_argument("--dataset-rows", type=int, default=1000)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prompt",
        default="Sparse optimization helps language models",
    )
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    if device.type == "cuda":
        print("gpu:", torch.cuda.get_device_name(0))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    model.config.pad_token_id = tokenizer.pad_token_id

    split = _dataset(tokenizer, args.dataset_rows, args.block_size)
    training_args = _training_args(args)
    optimizer, scheduler = get_scao_optimizer(
        model,
        training_args,
        scao_kwargs={
            "precond_freq": 20,
            "max_precond_dim": 1024,
            "k_min": 4,
            "k_max": 32,
            "async_precond": True,
            "noise_std_init": 0.0,
            "sparsity": 0.0,
            "lars_coeff": 0.0,
            "lookahead_k": 0,
        },
        num_training_steps=args.max_steps,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        optimizers=(optimizer, scheduler),
    )
    trainer.train()
    print(trainer.evaluate())

    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    output = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        pad_token_id=tokenizer.pad_token_id,
    )
    print(tokenizer.decode(output[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
