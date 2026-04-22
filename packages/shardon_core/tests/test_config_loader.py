from pathlib import Path

from shardon_core.config.loader import load_repository_config


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "config").exists():
            return parent
    raise RuntimeError("repository root with config/ not found")


def test_load_repository_config_reads_enabled_directories() -> None:
    repo_root = _repo_root()
    config = load_repository_config(repo_root / "config")
    assert config.global_config.instance_name.startswith("shardon")
    assert "mock-vllm-v1" in config.backends
    assert "demo-chat-model" in config.models
    assert config.models["demo-chat-model"].model_capabilities == ["text"]
    assert "chat-a" in config.deployments
    assert "group-a" in config.gpu_groups
    assert "gpu0" in config.gpu_devices


def test_enabled_directories_use_symlinks() -> None:
    repo_root = _repo_root()
    assert (repo_root / "config" / "backends-enabled" / "mock-vllm-v1.yaml").exists()
    assert (repo_root / "config" / "models-enabled" / "demo-chat-model.yaml").exists()
    assert (repo_root / "config" / "deployments-enabled" / "chat-a.yaml").exists()
