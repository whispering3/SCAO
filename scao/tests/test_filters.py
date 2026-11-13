import torch

from scao.filters import (
    _AdaptiveWarmupScheduler,
    _DynamicSparseFilter,
    _gsnr_clip,
    _LazyPrecondTrigger,
    _SparseGradFilter,
)


def test_sparse_filter_masks_low_magnitude_entries_and_updates_ema() -> None:
    filt = _SparseGradFilter(sparsity=0.5, ema=0.5)
    gradient = torch.tensor([1.0, 2.0, 3.0, 4.0])

    first = filt(gradient)
    second = filt(gradient * 2)

    assert first.shape == gradient.shape
    assert (first == 0).any()
    assert torch.isfinite(second).all()
    assert filt.mag_ema is not None


def test_dynamic_sparse_filter_keeps_sparsity_within_bounds() -> None:
    filt = _DynamicSparseFilter(base_sparsity=0.6, min_sparsity=0.4, max_sparsity=0.8)
    filt.set_global_norm_ref(2.0)

    output = filt(torch.tensor([1.0, 2.0, 3.0, 4.0]))

    assert output.shape == (4,)
    assert 0.4 <= filt.sparsity <= 0.8


def test_adaptive_warmup_exits_after_stable_gradients() -> None:
    scheduler = _AdaptiveWarmupScheduler(
        warmup_steps=20, stability_threshold=0.01, patience=2, min_warmup=1
    )

    assert scheduler.update(1, 1.0) is True
    assert scheduler.update(2, 1.0) is True
    assert scheduler.update(3, 1.0) is False
    assert scheduler.exited_early is True
    assert scheduler.actual_warmup_steps == 3


def test_gsnr_clip_masks_low_signal_entries() -> None:
    grad = torch.ones(2)
    exp_avg = torch.tensor([2.0, 0.1])
    exp_avg_sq = torch.ones(2)

    clipped = _gsnr_clip(grad, exp_avg, exp_avg_sq, eps=1e-8, clip_snr=0.5)

    assert torch.equal(clipped, torch.tensor([1.0, 0.0]))


def test_lazy_preconditioner_trigger_reacts_to_change_and_max_skip() -> None:
    trigger = _LazyPrecondTrigger(delta_threshold=0.1, max_skip=3)

    assert trigger.should_update(1.0) is False
    assert trigger.should_update(1.0) is False
    assert trigger.should_update(1.0) is True
    assert trigger.should_update(2.0) is True
