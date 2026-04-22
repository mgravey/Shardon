from pathlib import Path

from shardon_core.backends.base import ManagedProcess
from shardon_core.backends.registry import BackendRegistry
from shardon_core.config.schemas import (
    AdminUserRecord,
    BackendRuntimeConfig,
    DeploymentConfig,
    GPUDeviceConfig,
    GPUGroupConfig,
    GlobalConfig,
    ModelConfig,
    RepositoryConfig,
)
from shardon_core.logging.events import EventLogger


def _config() -> RepositoryConfig:
    return RepositoryConfig(
        global_config=GlobalConfig(),
        backends={
            "backend-a": BackendRuntimeConfig(
                id="backend-a",
                backend_type="vllm",
                version="1.0",
                display_name="backend-a",
                runtime_dir=".",
                working_directory=".",
                base_url="http://127.0.0.1:8100",
                launch_command=["python3", "-m", "http.server", "8100"],
                environment={"BASE": "1"},
                gpu_group_overrides={
                    "group-b": {
                        "base_url": "http://127.0.0.1:8101",
                        "launch_command": ["python3", "-m", "http.server", "8101"],
                        "environment": {"GROUP": "b"},
                    }
                },
            )
        },
        models={
            "model-a": ModelConfig(
                id="model-a",
                source="/model-a",
                display_name="Model A",
                backend_compatibility=["vllm"],
                tasks=["chat"],
            )
        },
        deployments={
            "dep-a": DeploymentConfig(
                id="dep-a",
                model_id="model-a",
                backend_runtime_id="backend-a",
                gpu_group_ids=["group-a", "group-b"],
                api_model_name="demo",
                display_name="Demo",
                tasks=["chat"],
            )
        },
        gpu_devices={
            "gpu0": GPUDeviceConfig(id="gpu0"),
            "gpu1": GPUDeviceConfig(id="gpu1"),
        },
        gpu_groups={
            "group-a": GPUGroupConfig(id="group-a", display_name="A", gpu_ids=["gpu0"]),
            "group-b": GPUGroupConfig(id="group-b", display_name="B", gpu_ids=["gpu1"]),
        },
        admin_users={
            "admin": AdminUserRecord(
                username="admin",
                password_hash="x",
                created_at="2026-04-21T00:00:00+00:00",
            )
        },
    )


def test_backend_registry_resolves_group_specific_overrides(tmp_path: Path) -> None:
    registry = BackendRegistry(_config(), tmp_path, EventLogger(tmp_path))
    resolved = registry.resolve_backend("backend-a", gpu_group_id="group-b")
    assert resolved.base_url == "http://127.0.0.1:8101"
    assert resolved.launch_command[-1] == "8101"
    assert resolved.environment["BASE"] == "1"
    assert resolved.environment["GROUP"] == "b"


def test_backend_registry_start_uses_selected_gpu_group(tmp_path: Path) -> None:
    config = _config()
    registry = BackendRegistry(config, tmp_path, EventLogger(tmp_path))
    captured: dict[str, object] = {}

    def fake_start(*, backend, deployment, extra_env=None):  # type: ignore[no-untyped-def]
        captured["base_url"] = backend.base_url
        captured["launch_command"] = backend.launch_command
        captured["extra_env"] = extra_env or {}
        return ManagedProcess(
            deployment_id=deployment.id,
            pid=4242,
            command=backend.launch_command,
            log_path=tmp_path / "dep-a.log",
            started_at="2026-04-22T00:00:00+00:00",
        )

    registry.supervisor.start = fake_start  # type: ignore[method-assign]
    pid = registry.ensure_started(config.deployments["dep-a"], gpu_group_id="group-b")
    assert pid == 4242
    assert captured["base_url"] == "http://127.0.0.1:8101"
    assert captured["launch_command"] == ["python3", "-m", "http.server", "8101"]
    assert captured["extra_env"]["SHARDON_GPU_GROUP_ID"] == "group-b"
