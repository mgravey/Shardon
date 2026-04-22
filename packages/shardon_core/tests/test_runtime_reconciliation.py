from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import time
from pathlib import Path

from shardon_core.gpu.provider import MockGPUProvider
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


def test_prepare_group_for_load_can_stop_adopted_process(tmp_path: Path) -> None:
    repo_root = _copy_repo_fixture(tmp_path)
    runtime = ShardonRuntime(repo_root=repo_root, gpu_provider=MockGPUProvider())
    process = subprocess.Popen(["python3", "-c", "import time; time.sleep(30)"])
    try:
        runtime.state_store.mutate(
            lambda snapshot: snapshot.model_copy(
                update={
                    "deployments": {
                        "chat-a": DeploymentRuntimeState(
                            deployment_id="chat-a",
                            gpu_group_id="group-a",
                            backend_runtime_id="mock-vllm-v1",
                            loaded=True,
                            process_id=process.pid,
                            resident_memory_fraction=0.9,
                        )
                    }
                }
            )
        )
        runtime._prepare_group_for_load(["chat-a"], reason="test switch")
        deadline = time.time() + 5
        while process.poll() is None and time.time() < deadline:
            time.sleep(0.05)
        assert process.poll() is not None
        assert runtime.snapshot().deployments["chat-a"].loaded is False
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def test_refresh_backend_health_deduplicates_repeated_failure_events(tmp_path: Path) -> None:
    repo_root = _copy_repo_fixture(tmp_path)
    runtime = ShardonRuntime(repo_root=repo_root, gpu_provider=MockGPUProvider())
    process = subprocess.Popen(["python3", "-c", "import time; time.sleep(30)"])
    try:
        runtime.state_store.mutate(
            lambda snapshot: snapshot.model_copy(
                update={
                    "deployments": {
                        "chat-a": DeploymentRuntimeState(
                            deployment_id="chat-a",
                            gpu_group_id="group-a",
                            backend_runtime_id="mock-vllm-v1",
                            loaded=True,
                            process_id=process.pid,
                            resident_memory_fraction=0.9,
                        )
                    }
                }
            )
        )

        async def failing_health(backend_runtime_id: str) -> dict[str, str]:
            raise RuntimeError("All connection attempts failed")

        runtime.backends.health = failing_health  # type: ignore[method-assign]
        asyncio.run(runtime.refresh_backend_health())
        asyncio.run(runtime.refresh_backend_health())

        events_path = repo_root / "state" / "events" / "events.jsonl"
        failed = []
        for line in events_path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("category") != "backend.health_failed":
                continue
            if entry.get("data", {}).get("backend_runtime_id") != "mock-vllm-v1":
                continue
            failed.append(entry)
        assert len(failed) == 1
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
