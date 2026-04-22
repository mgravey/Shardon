from shardon_core.config.schemas import GlobalConfig


def test_effective_timeouts_default_to_grace_plus_five_minutes() -> None:
    config = GlobalConfig()
    assert config.effective_interactive_request_timeout_seconds() == 600
    assert config.effective_backend_startup_timeout_seconds() == 600


def test_effective_timeouts_respect_explicit_values() -> None:
    config = GlobalConfig(
        switch_grace_window_seconds=900,
        interactive_request_timeout_seconds=1200,
        backend_startup_timeout_seconds=1500,
    )
    assert config.effective_interactive_request_timeout_seconds() == 1200
    assert config.effective_backend_startup_timeout_seconds() == 1500
