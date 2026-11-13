from scao.config import SCAOConfig


def test_lora_config_is_memory_bounded() -> None:
    config = SCAOConfig.for_lora()

    assert config.factor_mode == "wide"
    assert config.max_precond_dim == 256
    assert config.use_int8_ema is True


def test_config_applies_overrides() -> None:
    config = SCAOConfig.for_lora(k_max=16, precond_freq=20)

    assert config.to_kwargs()["k_max"] == 16
    assert config.to_kwargs()["precond_freq"] == 20
