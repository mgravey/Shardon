import asyncio
import shutil
from pathlib import Path

from shardon_core.gpu.provider import MockGPUProvider
from shardon_core.services.runtime import ShardonRuntime
from shardon_core.state.models import ActiveRequest, DeploymentRuntimeState, GPUProcessInfo


def _copy_repo_fixture(tmp_path: Path) -> Path:
    source_root = Path(__file__).resolve().parents[4]
    target_root = tmp_path / "repo"
    shutil.copytree(source_root / "config", target_root / "config")
    (target_root / "state").mkdir(parents=True, exist_ok=True)
    return target_root


def test_keep_free_kills_loaded_runtime_on_other_user_activity(tmp_path: Path) -> None:
    repo_root = _copy_repo_fixture(tmp_path)
    gpu_provider = MockGPUProvider()
    runtime = ShardonRuntime(repo_root=repo_root, gpu_provider=gpu_provider)
    runtime.state_store.mutate(
        lambda snapshot: snapshot.model_copy(
            update={
                "deployments": {
                    "chat-b": DeploymentRuntimeState(
                        deployment_id="chat-b",
                        gpu_group_id="group-b",
                        backend_runtime_id="mock-vllm-v2",
                        loaded=True,
                        process_id=1234,
                        resident_memory_fraction=0.9,
                    )
                }
            }
        )
    )
    gpu_provider.set_processes(
        [
            GPUProcessInfo(
                pid=9999,
                user_name="other-user",
                gpu_id="gpu1",
                command="python external.py",
                memory_mb=100,
            )
        ]
    )
    stopped: list[str] = []
    runtime.backends.stop = lambda deployment_id, force=False: stopped.append(deployment_id)  # type: ignore[method-assign]
    runtime.refresh_gpu_observations()
    snapshot = runtime.enforce_keep_free()
    assert stopped == ["chat-b"]
    assert snapshot.deployments["chat-b"].loaded is False
    assert snapshot.deployments["chat-b"].keep_free_killed_at is not None


def test_drain_force_kills_after_timeout(tmp_path: Path) -> None:
    repo_root = _copy_repo_fixture(tmp_path)
    runtime = ShardonRuntime(repo_root=repo_root, gpu_provider=MockGPUProvider())
    runtime.state_store.mutate(
        lambda snapshot: snapshot.model_copy(
            update={
                "deployments": {
                    "chat-b": DeploymentRuntimeState(
                        deployment_id="chat-b",
                        gpu_group_id="group-b",
                        backend_runtime_id="mock-vllm-v2",
                        loaded=True,
                        process_id=1234,
                        resident_memory_fraction=0.9,
                        active_request_ids=["req-1"],
                    )
                },
                "active_requests": {
                    "req-1": ActiveRequest(
                        id="req-1",
                        user_name="alice",
                        api_key_id="key-1",
                        deployment_id="chat-b",
                        backend_runtime_id="mock-vllm-v2",
                        gpu_group_id="group-b",
                        request_class="interactive",
                        model_name="demo-chat",
                        status="running",
                        priority=100,
                        created_at="2026-04-21T00:00:00+00:00",
                    )
                },
            }
        )
    )
    stopped: list[str] = []
    runtime.backends.stop = lambda deployment_id, force=False: stopped.append(deployment_id)  # type: ignore[method-assign]
    drain = asyncio.run(runtime.drain_group("group-b", 0))
    assert drain.status == "forced"
    assert stopped == ["chat-b"]

