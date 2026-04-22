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
        lambda deployment_id, gpu_group_id=None, force=False: stop_calls.append((deployment_id, gpu_group_id, force))
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


def test_load_deployment_falls_back_to_next_eligible_group_when_preferred_busy(tmp_path: Path) -> None:
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
