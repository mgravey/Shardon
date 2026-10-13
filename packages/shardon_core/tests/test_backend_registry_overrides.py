from pathlib import Path

from shardon_core.backends.base import ManagedProcess
from shardon_core.backends.registry import BackendRegistry
from shardon_core.config.schemas import (
    AdminUserRecord,
    BackendRuntimeConfig,
    DeploymentConfig,
    GlobalConfig,
    GPUDeviceConfig,
    GPUGroupConfig,
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


def test_backend_registry_appends_model_runtime_launch_args(tmp_path: Path) -> None:
    config = _config()
    config.models["model-a"].runtime_launch_args = ["--enable-prefix-caching"]
    config.models["model-a"].runtime_launch_args_by_backend_type = {
        "vllm": [
            "--tool-call-parser",
            "qwen3_coder",
            '--speculative-config={"method":"mtp","num_speculative_tokens":3}',
        ]
    }
    registry = BackendRegistry(config, tmp_path, EventLogger(tmp_path))
    captured: dict[str, object] = {}

    def fake_start(*, backend, deployment, extra_env=None):  # type: ignore[no-untyped-def]
        captured["launch_command"] = backend.launch_command
        captured["extra_env"] = extra_env or {}
        return ManagedProcess(
            deployment_id=deployment.id,
            pid=4343,
            command=backend.launch_command,
            log_path=tmp_path / "dep-a.log",
            started_at="2026-04-22T00:00:00+00:00",
        )

    registry.supervisor.start = fake_start  # type: ignore[method-assign]
    pid = registry.ensure_started(config.deployments["dep-a"], gpu_group_id="group-a")
    assert pid == 4343
    assert captured["launch_command"] == [
        "python3",
        "-m",
        "http.server",
        "8100",
        "--enable-prefix-caching",
        "--tool-call-parser",
        "qwen3_coder",
        '--speculative-config={"method":"mtp","num_speculative_tokens":3}',
    ]
    assert (
        captured["extra_env"]["SHARDON_MODEL_RUNTIME_LAUNCH_ARGS"]
        == '--enable-prefix-caching --tool-call-parser qwen3_coder '
        '--speculative-config={"method":"mtp","num_speculative_tokens":3}'
    )


def test_real_child_receives_uuid_selection_and_environment_precedence(tmp_path, monkeypatch):
    import json
    import sys

    from shardon_core.utils.env import load_dotenv_file

    config = _config()
    backend = config.backends["backend-a"]
    base_uuid = "GPU-00000000-0000-0000-0000-000000000001"
    group_uuid = "GPU-00000000-0000-0000-0000-000000000002"
    # Ordered multi-device selection must also be passed byte-for-byte.
    selection = f"{group_uuid},{base_uuid}"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-inherited-synthetic")
    monkeypatch.setenv("SHARDON_TEST_PARENT", "parent")
    monkeypatch.delenv("SHARDON_TEST_DOTENV", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "CUDA_VISIBLE_DEVICES=GPU-dotenv-synthetic\n"
        "SHARDON_TEST_PARENT=dotenv\nSHARDON_TEST_DOTENV=dotenv\n"
    )
    load_dotenv_file(env_file)
    keys = [
        "CUDA_VISIBLE_DEVICES",
        "SHARDON_TEST_PARENT",
        "SHARDON_TEST_DOTENV",
        "SHARDON_TEST_BASE",
        "SHARDON_TEST_SHARED",
        "SHARDON_GPU_GROUP_ID",
    ]
    command = [
        sys.executable,
        "-c",
        f"import json,os; print(json.dumps({{k:os.environ.get(k) for k in {keys!r}}}))",
    ]
    backend.launch_command = command
    backend.environment = {
        "CUDA_VISIBLE_DEVICES": base_uuid,
        "SHARDON_TEST_BASE": "base",
        "SHARDON_TEST_SHARED": "base",
        "SHARDON_GPU_GROUP_ID": "wrong-base",
    }
    override = backend.gpu_group_overrides["group-b"]
    override.launch_command = command
    override.environment = {
        "CUDA_VISIBLE_DEVICES": selection,
        "SHARDON_TEST_SHARED": "group",
        "SHARDON_GPU_GROUP_ID": "wrong-group",
    }
    registry = BackendRegistry(config, tmp_path, EventLogger(tmp_path))
    for group, expected_selection, shared in [
        ("group-b", selection, "group"),
        ("group-a", base_uuid, "base"),
    ]:
        registry.ensure_started(config.deployments["dep-a"], gpu_group_id=group)
        managed = registry.supervisor.processes["dep-a"]
        try:
            assert managed.process.wait(timeout=5) == 0
            payload = json.loads(managed.log_path.read_text().splitlines()[-1])
            assert payload == dict(
                zip(
                    keys,
                    [expected_selection, "parent", "dotenv", "base", shared, group],
                    strict=True,
                )
            )
        finally:
            registry.supervisor.kill("dep-a")
    # Resolving one group never contaminates the shared backend definition.
    assert backend.environment["CUDA_VISIBLE_DEVICES"] == base_uuid


def test_inventory_does_not_generate_cuda_environment(tmp_path, monkeypatch):
    import json
    import sys

    config = _config()
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    config.gpu_devices["gpu0"].uuid = "GPU-00000000-0000-0000-0000-000000000001"
    config.backends["backend-a"].launch_command = [
        sys.executable,
        "-c",
        "import os,json; print(json.dumps(os.getenv('CUDA_VISIBLE_DEVICES')))",
    ]
    registry = BackendRegistry(config, tmp_path, EventLogger(tmp_path))
    registry.ensure_started(config.deployments["dep-a"], gpu_group_id="group-a")
    managed = registry.supervisor.processes["dep-a"]
    try:
        assert managed.process.wait(timeout=5) == 0
        assert json.loads(managed.log_path.read_text()) is None
    finally:
        registry.supervisor.kill("dep-a")
