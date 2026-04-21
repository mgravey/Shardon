from pathlib import Path

from shardon_core.config.loader import load_repository_config


def test_load_repository_config_reads_enabled_directories() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    config = load_repository_config(repo_root / "config")
    assert config.global_config.instance_name == "shardon-demo"
    assert "mock-vllm-v1" in config.backends
    assert "demo-chat-model" in config.models
    assert "chat-a" in config.deployments
    assert "group-a" in config.gpu_groups
    assert "gpu0" in config.gpu_devices

