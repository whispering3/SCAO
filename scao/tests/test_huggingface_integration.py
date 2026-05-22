from __future__ import annotations

import inspect
import sys
import types

import pytest
import torch.nn as nn

from scao.integrations.huggingface import SCAOTrainer, get_scao_optimizer


class _Args:
    learning_rate = 1e-3
    warmup_steps = 0
    weight_decay = 0.01
    max_steps = 10
    lr_scheduler_type = "linear"


class _TinyPeftLikeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(8, 4)
        self.base = nn.Linear(4, 4)
        self.lora_A = nn.Linear(4, 2, bias=False)
        self.lora_B = nn.Linear(2, 4, bias=False)
        self.lm_head = nn.Linear(4, 8, bias=False)


def _install_transformers_stub(monkeypatch) -> None:
    module = types.ModuleType("transformers")

    def get_scheduler(**kwargs):
        return {"scheduler_kwargs": kwargs}

    module.get_scheduler = get_scheduler
    monkeypatch.setitem(sys.modules, "transformers", module)


def _group_param_ids(opt, enabled: bool) -> set[int]:
    ids: set[int] = set()
    for group in opt.param_groups:
        if group["preconditioner_enabled"] is enabled:
            ids.update(id(param) for param in group["params"])
    return ids


def test_hf_auto_policy_skips_embeddings_and_lm_head(monkeypatch):
    _install_transformers_stub(monkeypatch)
    model = _TinyPeftLikeModel()

    opt, scheduler = get_scao_optimizer(
        model,
        _Args(),
        preconditioner_policy="auto",
        num_training_steps=10,
    )

    enabled = _group_param_ids(opt, True)
    disabled = _group_param_ids(opt, False)

    assert id(model.base.weight) in enabled
    assert id(model.lora_A.weight) in enabled
    assert id(model.embed_tokens.weight) in disabled
    assert id(model.lm_head.weight) in disabled
    assert scheduler is not None


def test_hf_default_policy_preconditions_all_trainable_params(monkeypatch):
    _install_transformers_stub(monkeypatch)
    model = _TinyPeftLikeModel()

    opt, _ = get_scao_optimizer(model, _Args(), num_training_steps=10)

    enabled = _group_param_ids(opt, True)
    assert id(model.base.weight) in enabled
    assert id(model.lora_A.weight) in enabled
    assert id(model.embed_tokens.weight) in enabled
    assert id(model.lm_head.weight) in enabled


def test_hf_adapters_only_policy_preconditions_only_lora(monkeypatch):
    _install_transformers_stub(monkeypatch)
    model = _TinyPeftLikeModel()

    opt, _ = get_scao_optimizer(
        model,
        _Args(),
        preconditioner_policy="adapters_only",
        num_training_steps=10,
    )

    enabled = _group_param_ids(opt, True)
    disabled = _group_param_ids(opt, False)

    assert id(model.lora_A.weight) in enabled
    assert id(model.lora_B.weight) in enabled
    assert id(model.base.weight) in disabled
    assert id(model.embed_tokens.weight) in disabled


def test_hf_invalid_preconditioner_policy_raises(monkeypatch):
    _install_transformers_stub(monkeypatch)
    model = _TinyPeftLikeModel()

    try:
        get_scao_optimizer(model, _Args(), preconditioner_policy="surprise")
    except ValueError as exc:
        assert "preconditioner_policy" in str(exc)
    else:
        raise AssertionError("expected invalid preconditioner_policy to raise")


def test_scao_trainer_exposes_preconditioner_policy_args():
    if SCAOTrainer is None:
        pytest.skip("transformers is not installed")

    signature = inspect.signature(SCAOTrainer.__init__)
    assert "preconditioner_policy" in signature.parameters
    assert "no_precond_names" in signature.parameters
    assert "adapter_names" in signature.parameters
