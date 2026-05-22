# mypy: ignore-errors
"""
HuggingFace Transformers integration for SCAO
==============================================

Provides three integration paths:

Path A — Manual (recommended for control)
-----------------------------------------
    from scao import SCAO
    from scao.integrations.huggingface import get_scao_optimizer

    optimizer, scheduler = get_scao_optimizer(model, training_args)
    trainer = Trainer(model=model, args=training_args,
                      optimizers=(optimizer, scheduler))

Path B — SCAOTrainer drop-in
-----------------------------
    from scao.integrations.huggingface import SCAOTrainer

    trainer = SCAOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        scao_kwargs=dict(precond_freq=20, k_min=8, k_max=64),
    )
    trainer.train()

Path C — Trainer callback for monitoring
-----------------------------------------
    from scao.integrations.huggingface import SCAOMonitorCallback
    from scao import SCAO

    optimizer = SCAO(model.parameters(), lr=training_args.learning_rate)
    trainer = Trainer(model=model, args=training_args,
                      optimizers=(optimizer, None))
    trainer.add_callback(SCAOMonitorCallback(optimizer))
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch.nn as nn

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Path A: get_scao_optimizer
# ---------------------------------------------------------------------------

def get_scao_optimizer(
    model: nn.Module,
    training_args,
    scao_kwargs: dict | None = None,
    no_decay_names: tuple[str, ...] = ("bias", "LayerNorm.weight", "layer_norm.weight"),
    no_precond_names: tuple[str, ...] = (
        "embed",
        "embedding",
        "embed_tokens",
        "lm_head",
        "wte",
        "wpe",
    ),
    adapter_names: tuple[str, ...] = (
        "adapter",
        "lora",
        "ia3",
        "prefix",
        "prompt",
    ),
    preconditioner_policy: str = "all",
    num_training_steps: int | None = None,
):
    """
    Build a SCAO optimizer + learning-rate scheduler compatible with
    HuggingFace ``Trainer``.

    Applies weight decay only to parameters whose names do NOT match
    ``no_decay_names`` (standard HF practice).

    Args:
        model: the model to optimise
        training_args: ``transformers.TrainingArguments`` instance
        scao_kwargs: additional SCAO hyperparameters (override defaults)
        no_decay_names: parameter name substrings exempt from weight decay
        no_precond_names: parameter name substrings that should skip SCAO
            preconditioning while still using the optimizer moments. This is
            useful for large embedding/output matrices in 100B+ models.
        adapter_names: parameter name substrings used to identify PEFT/adapter
            parameters for ``preconditioner_policy="adapters_only"``.
        preconditioner_policy: one of:
            ``"all"``: precondition all trainable parameters. This is the
                backwards-compatible default.
            ``"auto"``: precondition trainable non-embedding parameters.
            ``"none"``: disable preconditioning for all parameters.
            ``"adapters_only"``: precondition only adapter/LoRA-style params.
        num_training_steps: explicit scheduler length. Required for an accurate
            scheduler when ``training_args.max_steps`` is not set.

    Returns:
        (optimizer, scheduler) tuple ready for ``Trainer(optimizers=...)``.
        The scheduler is ``None`` when the total step count is unknown.

    Example::

        from transformers import TrainingArguments, Trainer
        from scao.integrations.huggingface import get_scao_optimizer

        args = TrainingArguments(output_dir="out", num_train_epochs=3, ...)
        optimizer, scheduler = get_scao_optimizer(model, args)
        trainer = Trainer(model=model, args=args,
                          optimizers=(optimizer, scheduler))
    """
    try:
        from transformers import get_scheduler  # type: ignore[import]
    except ImportError as e:
        raise ImportError(
            "transformers is required for get_scao_optimizer. "
            "Install it with: pip install transformers"
        ) from e

    from scao.optimizer import SCAO

    valid_policies = {"auto", "all", "none", "adapters_only"}
    if preconditioner_policy not in valid_policies:
        raise ValueError(
            f"Invalid preconditioner_policy={preconditioner_policy!r}. "
            f"Expected one of {sorted(valid_policies)}."
        )

    def should_precondition(name: str) -> bool:
        if preconditioner_policy == "all":
            return True
        if preconditioner_policy == "none":
            return False
        if preconditioner_policy == "adapters_only":
            return any(token in name for token in adapter_names)
        return not any(token in name for token in no_precond_names)

    # Separate parameters with / without weight decay (standard HF split)
    decay_precond_params, decay_plain_params = [], []
    no_decay_precond_params, no_decay_plain_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        use_decay = not any(nd in name for nd in no_decay_names)
        use_precond = should_precondition(name)
        if use_decay and use_precond:
            decay_precond_params.append(param)
        elif use_decay:
            decay_plain_params.append(param)
        elif use_precond:
            no_decay_precond_params.append(param)
        else:
            no_decay_plain_params.append(param)

    param_groups = []
    if decay_precond_params:
        param_groups.append({
            "params": decay_precond_params,
            "weight_decay": training_args.weight_decay,
            "preconditioner_enabled": True,
        })
    if decay_plain_params:
        param_groups.append({
            "params": decay_plain_params,
            "weight_decay": training_args.weight_decay,
            "preconditioner_enabled": False,
        })
    if no_decay_precond_params:
        param_groups.append({
            "params": no_decay_precond_params,
            "weight_decay": 0.0,
            "preconditioner_enabled": True,
        })
    if no_decay_plain_params:
        param_groups.append({
            "params": no_decay_plain_params,
            "weight_decay": 0.0,
            "preconditioner_enabled": False,
        })

    defaults: dict[str, Any] = {
        "lr": training_args.learning_rate,
        "warmup_steps": int(getattr(training_args, "warmup_steps", 100)),
    }
    if scao_kwargs:
        defaults.update(scao_kwargs)

    optimizer = SCAO(param_groups, **defaults)

    # Build HF-compatible LR scheduler only when the total length is known.
    # A hard-coded fallback silently distorts cosine/linear schedules on most
    # real runs, so manual users must pass num_training_steps or max_steps.
    if num_training_steps is None:
        max_steps = int(getattr(training_args, "max_steps", -1))
        num_training_steps = max_steps if max_steps > 0 else None

    if num_training_steps is None:
        return optimizer, None

    scheduler = get_scheduler(
        name=getattr(training_args, "lr_scheduler_type", "cosine"),
        optimizer=optimizer,
        num_warmup_steps=getattr(training_args, "warmup_steps", 0),
        num_training_steps=num_training_steps,
    )

    return optimizer, scheduler


# ---------------------------------------------------------------------------
# Path B: SCAOTrainer
# ---------------------------------------------------------------------------

def _make_scao_trainer_class():
    """
    Lazily build SCAOTrainer to avoid importing transformers at module load.
    """
    try:
        from transformers import Trainer  # type: ignore[import]
    except ImportError:
        return None

    class SCAOTrainer(Trainer):
        """
        Drop-in ``Trainer`` subclass that uses SCAO as the optimizer.

        Args:
            scao_kwargs: dict of extra SCAO hyperparameters (e.g.
                ``precond_freq``, ``k_min``, ``k_max``, ``tau``).

        All other arguments are passed through to ``transformers.Trainer``.

        Example::

            trainer = SCAOTrainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                scao_kwargs=dict(precond_freq=20, k_max=64),
            )
            trainer.train()
        """

        def __init__(
            self,
            *args,
            scao_kwargs: dict | None = None,
            preconditioner_policy: str = "all",
            no_precond_names: tuple[str, ...] = (
                "embed",
                "embedding",
                "embed_tokens",
                "lm_head",
                "wte",
                "wpe",
            ),
            adapter_names: tuple[str, ...] = (
                "adapter",
                "lora",
                "ia3",
                "prefix",
                "prompt",
            ),
            **kwargs,
        ):
            self._scao_kwargs = scao_kwargs or {}
            self._scao_preconditioner_policy = preconditioner_policy
            self._scao_no_precond_names = no_precond_names
            self._scao_adapter_names = adapter_names
            super().__init__(*args, **kwargs)

        def create_optimizer(self):
            if self.optimizer is None:
                optimizer, _ = get_scao_optimizer(
                    self.model,
                    self.args,
                    scao_kwargs=self._scao_kwargs,
                    no_precond_names=self._scao_no_precond_names,
                    adapter_names=self._scao_adapter_names,
                    preconditioner_policy=self._scao_preconditioner_policy,
                )
                self.optimizer = optimizer
            return self.optimizer

        def create_optimizer_and_scheduler(self, num_training_steps: int):
            self.create_optimizer()
            if self.lr_scheduler is None:
                try:
                    from transformers import get_scheduler  # type: ignore[import]
                    self.lr_scheduler = get_scheduler(
                        name=self.args.lr_scheduler_type,
                        optimizer=self.optimizer,
                        num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                        num_training_steps=num_training_steps,
                    )
                except Exception:
                    super().create_optimizer_and_scheduler(num_training_steps)

    return SCAOTrainer


# Expose SCAOTrainer — will be None if transformers not installed
SCAOTrainer = _make_scao_trainer_class()


# ---------------------------------------------------------------------------
# Path C: SCAOMonitorCallback
# ---------------------------------------------------------------------------

def _make_monitor_callback():
    try:
        from transformers import TrainerCallback  # type: ignore[import]
    except ImportError:
        return None

    class SCAOMonitorCallback(TrainerCallback):
        """
        HuggingFace ``TrainerCallback`` that logs SCAO rank and curvature
        health metrics to the Trainer's log at every logging step.

        Args:
            optimizer: the SCAO optimizer instance
            log_every: override log frequency (default: use Trainer's setting)

        Example::

            trainer.add_callback(SCAOMonitorCallback(optimizer))
        """

        def __init__(self, optimizer, log_every: int | None = None) -> None:
            self.optimizer = optimizer
            self.log_every = log_every

        def on_log(self, args, state, control, logs=None, **kwargs):
            from scao.logging import collect_metrics
            metrics = collect_metrics(self.optimizer)
            if logs is not None:
                # Inject numeric metrics into the Trainer's log dict
                logs.update(
                    {k: v for k, v in metrics.items()
                     if isinstance(v, (int, float)) and k != "step"}
                )

    return SCAOMonitorCallback


SCAOMonitorCallback = _make_monitor_callback()
