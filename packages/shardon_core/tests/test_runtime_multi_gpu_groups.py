from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

from shardon_core.gpu.provider import MockGPUProvider
from shardon_core.scheduler.engine import SchedulerEngine
from shardon_core.services.runtime import ShardonRuntime
from shardon_core.state.models import DeploymentRuntimeState


def _source_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "config").exists():
            return parent
    raise RuntimeError("repository root with config/ not found")


def _copy_repo_fixture(tmp_path: Path) -> Path:
    source_root = _source_repo_root()
    target_root = tmp_path / "repo"
    shutil.copytree(source_root / "config", target_root / "config")
    for directory_name in ("admins-available", "admins-enabled"):
        admin_dir = target_root / "config" / "auth" / directory_name
        for admin_user in admin_dir.glob("*.yaml"):
            lines = []
            for line in admin_user.read_text(encoding="utf-8").splitlines():
                if line.startswith("created_at: "):
                    created_at = line.removeprefix("created_at: ").strip()
                    line = f'created_at: "{created_at}"'
                lines.append(line)
            admin_user.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (target_root / "state").mkdir(parents=True, exist_ok=True)
    return target_root


def _make_runtime(tmp_path: Path) -> ShardonRuntime:
    runtime = ShardonRuntime(repo_root=_copy_repo_fixture(tmp_path), gpu_provider=MockGPUProvider())
    deployment = runtime.config.deployments["chat-a"]
    runtime.config.deployments["chat-a"] = deployment.model_copy(
        update={"gpu_group_ids": ["group-a", "group-b"], "gpu_group_id": "group-a"}
    )
    runtime.scheduler = SchedulerEngine(runtime.config)
    return runtime


def test_load_deployment_can_switch_selected_gpu_group(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    start_calls: list[tuple[str, str]] = []
    stop_calls: list[tuple[str, str | None, bool]] = []

    async def fake_start_and_ready(deployment, *, gpu_group_id: str):  # type: ignore[no-untyped-def]
        start_calls.append((deployment.id, gpu_group_id))
        return {
            "deployment_id": deployment.id,
            "backend_runtime_id": deployment.backend_runtime_id,
            "gpu_group_id": gpu_group_id,
            "pid": 1_000 + len(start_calls),
            "attempt_count": 1,
            "ready_at": "2026-04-22T00:00:00+00:00",
            "payload": {"status": "ok"},
        }

    runtime.backends.ensure_started_and_ready = fake_start_and_ready  # type: ignore[method-assign]
    runtime.backends.stop = (  # type: ignore[method-assign]
        lambda deployment_id, gpu_group_id=None, force=False: stop_calls.append(
            (deployment_id, gpu_group_id, force)
        )
    )

    first = asyncio.run(runtime.load_deployment(deployment_id="chat-a", actor="test"))
    assert first["selected_gpu_group_id"] == "group-a"
    assert runtime.snapshot().deployments["chat-a"].selected_gpu_group_id == "group-a"

    second = asyncio.run(
        runtime.load_deployment(deployment_id="chat-a", gpu_group_id="group-b", actor="test")
    )
    assert second["selected_gpu_group_id"] == "group-b"
    assert runtime.snapshot().deployments["chat-a"].selected_gpu_group_id == "group-b"
    assert start_calls == [("chat-a", "group-a"), ("chat-a", "group-b")]
    assert stop_calls[0][0] == "chat-a"
    assert stop_calls[0][1] == "group-a"


def test_load_deployment_falls_back_to_next_eligible_group_when_preferred_busy(
    tmp_path: Path,
) -> None:
    runtime = _make_runtime(tmp_path)
    start_calls: list[tuple[str, str]] = []

    async def fake_start_and_ready(deployment, *, gpu_group_id: str):  # type: ignore[no-untyped-def]
        start_calls.append((deployment.id, gpu_group_id))
        return {
            "deployment_id": deployment.id,
            "backend_runtime_id": deployment.backend_runtime_id,
            "gpu_group_id": gpu_group_id,
            "pid": 2_000 + len(start_calls),
            "attempt_count": 1,
            "ready_at": "2026-04-22T00:00:00+00:00",
            "payload": {"status": "ok"},
        }

    runtime.backends.ensure_started_and_ready = fake_start_and_ready  # type: ignore[method-assign]
    runtime.state_store.mutate(
        lambda snapshot: snapshot.model_copy(
            update={
                "deployments": {
                    "embed-a": DeploymentRuntimeState(
                        deployment_id="embed-a",
                        gpu_group_id="group-a",
                        selected_gpu_group_id="group-a",
                        backend_runtime_id="mock-sglang",
                        loaded=True,
                        process_id=3333,
                        resident_memory_fraction=0.35,
                        active_request_ids=["req-busy"],
                    )
                }
            }
        )
    )

    result = asyncio.run(runtime.load_deployment(deployment_id="chat-a", actor="test"))
    assert result["selected_gpu_group_id"] == "group-b"
    assert runtime.snapshot().deployments["chat-a"].selected_gpu_group_id == "group-b"
    assert start_calls == [("chat-a", "group-b")]


def test_gpu_refresh_drops_disappeared_devices_and_reports_missing_group(tmp_path, monkeypatch):
    import pytest
    from shardon_core.services.runtime import RuntimeOperationError

    runtime = _make_runtime(tmp_path)
    assert set(runtime.refresh_gpu_observations().gpu_observations) == {"gpu0", "gpu1"}
    remaining = runtime.gpu_provider.observe(runtime.config.gpu_devices)
    del remaining["gpu0"]
    monkeypatch.setattr(runtime.gpu_provider, "observe", lambda devices: remaining)
    snapshot = runtime.refresh_gpu_observations()
    assert set(snapshot.gpu_observations) == {"gpu1"}
    assert runtime._gpu_group_summary("group-a", snapshot)["unavailable_gpu_ids"] == ["gpu0"]
    with pytest.raises(RuntimeOperationError) as error:
        asyncio.run(runtime.load_deployment(deployment_id="chat-a", gpu_group_id="group-a"))
    assert "group-a: gpu0" in error.value.detail["reason"]
    assert not runtime.backends.supervisor.processes
    monkeypatch.setattr(runtime.gpu_provider, "observe", lambda devices: {})
    assert runtime.refresh_gpu_observations().gpu_observations == {}


def test_device_disappearing_after_admission_prevents_launch(tmp_path, monkeypatch):
    import pytest
    from shardon_core.services.runtime import RuntimeOperationError

    runtime = _make_runtime(tmp_path)
    available = runtime.gpu_provider.observe(runtime.config.gpu_devices)
    probes = iter([available, {}])
    monkeypatch.setattr(runtime.gpu_provider, "observe", lambda devices: next(probes))
    with pytest.raises(RuntimeOperationError) as error:
        asyncio.run(runtime.load_deployment(deployment_id="chat-a"))
    assert error.value.detail["gpu_group_id"] == "group-a"
    assert error.value.detail["readiness"]["unavailable_gpu_ids"] == ["gpu0"]
    state = runtime.snapshot().deployments["chat-a"]
    assert state.state == "failed"
    assert "gpu0" in state.last_error
    assert not runtime.backends.supervisor.processes


def test_start_failure_is_persisted_with_log_path(tmp_path):
    import pytest
    from shardon_core.services.runtime import RuntimeOperationError

    runtime = _make_runtime(tmp_path)
    backend = runtime.config.backends[runtime.config.deployments["chat-a"].backend_runtime_id]
    backend.working_directory = str(tmp_path)
    backend.launch_command = [str(tmp_path / "nonexistent-synthetic-backend")]
    with pytest.raises(RuntimeOperationError) as error:
        asyncio.run(runtime.load_deployment(deployment_id="chat-a"))
    detail = error.value.detail["readiness"]
    assert "FileNotFoundError" in detail["error"]
    assert Path(detail["log_path"]).is_file()
    state = runtime.snapshot().deployments["chat-a"]
    assert state.state == "failed"
    assert state.last_readiness_detail == detail
    assert any('"category": "backend.start_failed"' in event for event in runtime.read_events())


def test_early_backend_exit_preserves_exit_code_and_operator_logs(tmp_path, monkeypatch):
    import sys

    import pytest
    from shardon_core.services.runtime import RuntimeOperationError

    runtime = _make_runtime(tmp_path)
    backend = runtime.config.backends[runtime.config.deployments["chat-a"].backend_runtime_id]
    backend.working_directory = str(tmp_path)
    backend.launch_command = [
        sys.executable,
        "-c",
        "import sys; print('synthetic CUDA device unavailable', flush=True); "
        "sys.stderr.buffer.write(b'error details\\xff\\n'); sys.exit(23)",
    ]
    backend.startup_timeout_seconds = 3
    backend.readiness_poll_interval_seconds = 0.01

    async def unavailable_health(*args, **kwargs):
        raise ConnectionError("synthetic backend unavailable")

    monkeypatch.setattr(runtime.backends, "health", unavailable_health)
    with pytest.raises(RuntimeOperationError) as error:
        asyncio.run(runtime.load_deployment(deployment_id="chat-a"))
    detail = error.value.detail["readiness"]
    assert detail["returncode"] == 23
    assert detail["error"] == "backend process exited before readiness"
    assert detail["gpu_group_id"] == "group-a"
    assert runtime.snapshot().deployments["chat-a"].state == "failed"
    assert runtime.read_backend_log("chat-a") == [
        "synthetic CUDA device unavailable",
        "error details�",
    ]
    assert any('"returncode": 23' in event for event in runtime.read_events())
    assert not runtime.backends.supervisor.processes


def test_explicit_demo_provider_and_default_real_provider(tmp_path):
    from shardon_core.gpu.provider import NvidiaSMIProvider

    root = _copy_repo_fixture(tmp_path)
    runtime = ShardonRuntime(repo_root=root)
    assert isinstance(runtime.gpu_provider, MockGPUProvider)
    assert set(runtime.refresh_gpu_observations().gpu_observations) == {"gpu0", "gpu1"}
    router = root / "config" / "router.yaml"
    router.write_text(router.read_text().replace("gpu_provider: mock\n", ""))
    real_runtime = ShardonRuntime(repo_root=root)
    assert isinstance(real_runtime.gpu_provider, NvidiaSMIProvider)


def test_scheduler_fallback_reaches_child_and_group_specific_readiness(tmp_path, monkeypatch):
    import json
    import sys

    from shardon_core.config.schemas import BackendRuntimeOverride

    runtime = _make_runtime(tmp_path)
    available = runtime.gpu_provider.observe(runtime.config.gpu_devices)
    del available["gpu0"]
    monkeypatch.setattr(runtime.gpu_provider, "observe", lambda devices: available)
    backend = runtime.config.backends[runtime.config.deployments["chat-a"].backend_runtime_id]
    selected_uuid = "GPU-00000000-0000-0000-0000-000000000002"
    backend.environment = {"CUDA_VISIBLE_DEVICES": "GPU-00000000-0000-0000-0000-000000000001"}
    backend.gpu_group_overrides["group-b"] = BackendRuntimeOverride(
        environment={"CUDA_VISIBLE_DEVICES": selected_uuid},
        working_directory=str(tmp_path),
        launch_command=[
            sys.executable,
            "-c",
            "import json,os,time; "
            "print(json.dumps([os.getenv('CUDA_VISIBLE_DEVICES'), "
            "os.getenv('SHARDON_GPU_GROUP_ID')]), flush=True); time.sleep(10)",
        ],
        base_url="http://group-b.test:8101",
        health_path="/ready-b",
    )
    health_groups = []

    async def health(backend_runtime_id, *, gpu_group_id):
        adapter = runtime.backends.adapter_for(backend_runtime_id, gpu_group_id=gpu_group_id)
        assert adapter.backend.base_url == "http://group-b.test:8101"
        assert adapter.backend.health_path == "/ready-b"
        health_groups.append(gpu_group_id)
        # Use the child's output as the synthetic readiness signal.
        for _ in range(100):
            lines = runtime.read_backend_log("chat-a")
            if lines:
                return {"child_environment": json.loads(lines[-1])}
            await asyncio.sleep(0.01)
        raise AssertionError("synthetic child did not report readiness")

    monkeypatch.setattr(runtime.backends, "health", health)
    try:
        result = asyncio.run(runtime.load_deployment(deployment_id="chat-a"))
        state = runtime.snapshot().deployments["chat-a"]
        assert result["selected_gpu_group_id"] == state.selected_gpu_group_id == "group-b"
        assert state.last_readiness_detail["payload"]["child_environment"] == [
            selected_uuid,
            "group-b",
        ]
        assert health_groups == ["group-b"]
    finally:
        runtime.backends.supervisor.kill("chat-a")


def test_concurrent_group_load_cannot_relabel_existing_process(tmp_path):
    import pytest
    from shardon_core.services.runtime import RuntimeOperationError

    runtime = _make_runtime(tmp_path)
    starts = []

    async def run():
        first_start_entered = asyncio.Event()
        allow_first_start = asyncio.Event()

        async def start(deployment, *, gpu_group_id):
            starts.append(gpu_group_id)
            first_start_entered.set()
            await allow_first_start.wait()
            return {
                "pid": 4242,
                "ready_at": "2026-01-01T00:00:00+00:00",
                "gpu_group_id": gpu_group_id,
            }

        runtime.backends.ensure_started_and_ready = start
        first = asyncio.create_task(
            runtime.load_deployment(deployment_id="chat-a", gpu_group_id="group-a")
        )
        await first_start_entered.wait()
        second = asyncio.create_task(
            runtime.load_deployment(deployment_id="chat-a", gpu_group_id="group-b")
        )
        # Both decisions see the first deployment as starting, not yet loaded.
        await asyncio.sleep(0)
        allow_first_start.set()
        await first
        with pytest.raises(RuntimeOperationError, match="placement changed"):
            await second

    asyncio.run(run())
    state = runtime.snapshot().deployments["chat-a"]
    assert state.selected_gpu_group_id == "group-a"
    assert state.process_id == 4242
    assert starts == ["group-a"]
