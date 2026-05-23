from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import pytest

from shardon_core.api.schemas import ChatCompletionRequest
from shardon_core.auth.service import AuthResult
from shardon_core.gpu.provider import MockGPUProvider
from shardon_core.services.runtime import RuntimeOperationError, ShardonRuntime
from shardon_core.state.models import ActiveRequest, BatchJobState, DeploymentRuntimeState


def _source_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "config").exists():
            return parent
    raise RuntimeError("repository root with config/ not found")


def _copy_repo_fixture(tmp_path: Path) -> Path:
    source_root = _source_repo_root()
    target_root = tmp_path / "repo"
    shutil.copytree(source_root / "config", target_root / "config")
    router_config = target_root / "config" / "router.yaml"
    router_config.write_text(
        "\n".join(
            "state_root: state" if line.startswith("state_root: ") else line
            for line in router_config.read_text(encoding="utf-8").splitlines()
        )
        + "\n",
        encoding="utf-8",
    )
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


def _queued_request(request_id: str) -> ActiveRequest:
    return ActiveRequest(
        id=request_id,
        user_name="alice",
        api_key_id="k1",
        deployment_id="",
        backend_runtime_id="",
        gpu_group_id="",
        request_class="interactive",
        model_name="demo-chat",
        status="queued",
        priority=100,
        created_at="2026-04-21T00:00:00+00:00",
    )


def test_clear_queue_clears_interactive_queue(tmp_path: Path) -> None:
    repo_root = _copy_repo_fixture(tmp_path)
    runtime = ShardonRuntime(repo_root=repo_root, gpu_provider=MockGPUProvider())
    runtime.state_store.mutate(
        lambda snapshot: snapshot.model_copy(
            update={
                "queued_requests": [_queued_request("req-1"), _queued_request("req-2")],
            }
        )
    )
    result = runtime.clear_queue(clear_interactive=True, clear_batches=False, actor="test")
    snapshot = runtime.snapshot()
    assert result["cleared_interactive_requests"] == 2
    assert result["cancelled_batch_jobs"] == 0
    assert result["interactive_request_ids"] == ["req-1", "req-2"]
    assert snapshot.queued_requests == []


def test_clear_queue_can_cancel_queued_batches(tmp_path: Path) -> None:
    repo_root = _copy_repo_fixture(tmp_path)
    runtime = ShardonRuntime(repo_root=repo_root, gpu_provider=MockGPUProvider())
    runtime.state_store.mutate(
        lambda snapshot: snapshot.model_copy(
            update={
                "batch_jobs": {
                    "batch-1": BatchJobState(
                        id="batch-1",
                        api_key_id="k1",
                        user_name="alice",
                        model_name="demo-chat",
                        status="queued",
                        created_at="2026-04-21T00:00:00+00:00",
                        updated_at="2026-04-21T00:00:00+00:00",
                        total_items=2,
                    ),
                    "batch-2": BatchJobState(
                        id="batch-2",
                        api_key_id="k2",
                        user_name="bob",
                        model_name="demo-chat",
                        status="running",
                        created_at="2026-04-21T00:00:00+00:00",
                        updated_at="2026-04-21T00:00:00+00:00",
                        total_items=2,
                    ),
                }
            }
        )
    )
    result = runtime.clear_queue(clear_interactive=False, clear_batches=True, actor="test")
    snapshot = runtime.snapshot()
    assert result["cleared_interactive_requests"] == 0
    assert result["cancelled_batch_jobs"] == 1
    assert result["batch_job_ids"] == ["batch-1"]
    assert snapshot.batch_jobs["batch-1"].status == "cancelled"
    assert snapshot.batch_jobs["batch-2"].status == "running"


def test_unsupported_model_rejected_immediately_without_queueing(tmp_path: Path) -> None:
    repo_root = _copy_repo_fixture(tmp_path)
    runtime = ShardonRuntime(repo_root=repo_root, gpu_provider=MockGPUProvider())
    auth = AuthResult(
        id="k1",
        user_name="alice",
        priority=100,
        permissions=["inference"],
    )
    with pytest.raises(RuntimeOperationError) as exc:
        asyncio.run(
            runtime.route_chat(
                ChatCompletionRequest(
                    model="missing-model",
                    messages=[{"role": "user", "content": "ping"}],
                ),
                auth,
            )
        )
    assert exc.value.status_code == 404
    assert exc.value.detail["error"] == "no compatible deployment"
    assert exc.value.detail["model_name"] == "missing-model"
    snapshot = runtime.snapshot()
    assert snapshot.queued_requests == []


def _loaded_deployment_state(runtime: ShardonRuntime, deployment_id: str) -> DeploymentRuntimeState:
    deployment = runtime.config.deployments[deployment_id]
    gpu_group_id = deployment.preferred_gpu_group_id()
    return DeploymentRuntimeState(
        deployment_id=deployment.id,
        gpu_group_id=gpu_group_id,
        selected_gpu_group_id=gpu_group_id,
        backend_runtime_id=deployment.backend_runtime_id,
        loaded=True,
        state="ready",
        desired_state="loaded",
        current_model_name=deployment.api_model_name,
        loaded_at="2026-04-21T00:00:00+00:00",
    )


def _first_compatible_deployment(runtime: ShardonRuntime, task: str, capability: str):
    for deployment in runtime.config.deployments.values():
        if not deployment.enabled:
            continue
        if task not in deployment.tasks:
            continue
        if capability not in runtime._deployment_effective_capabilities(deployment):
            continue
        return deployment
    raise AssertionError(f"no fixture deployment supports {task}/{capability}")


def test_list_api_models_includes_currently_loaded_alias(tmp_path: Path) -> None:
    repo_root = _copy_repo_fixture(tmp_path)
    runtime = ShardonRuntime(repo_root=repo_root, gpu_provider=MockGPUProvider())
    deployment = _first_compatible_deployment(runtime, "chat", "text")
    runtime.state_store.mutate(
        lambda snapshot: snapshot.model_copy(
            update={
                "deployments": {
                    deployment.id: _loaded_deployment_state(runtime, deployment.id),
                },
            }
        )
    )

    models = {item["id"]: item for item in runtime.list_api_models()}

    assert "currently-loaded" in models
    assert models["currently-loaded"]["resolved_model_id"] == deployment.api_model_name
    assert models["currently-loaded"]["current_model_name"] == deployment.api_model_name
    assert models["currently-loaded"]["resolved_deployment_id"] == deployment.id


def test_currently_loaded_alias_resolves_to_loaded_compatible_deployment(tmp_path: Path) -> None:
    repo_root = _copy_repo_fixture(tmp_path)
    runtime = ShardonRuntime(repo_root=repo_root, gpu_provider=MockGPUProvider())
    deployment = _first_compatible_deployment(runtime, "chat", "text")
    runtime.state_store.mutate(
        lambda snapshot: snapshot.model_copy(
            update={
                "deployments": {
                    deployment.id: _loaded_deployment_state(runtime, deployment.id),
                },
            }
        )
    )

    resolved = runtime._resolve_model_alias(
        model_name="currently-loaded",
        task="chat",
        required_capability="text",
        snapshot=runtime.snapshot(),
    )

    assert resolved == deployment.api_model_name


def test_currently_loaded_alias_requires_compatible_loaded_deployment(tmp_path: Path) -> None:
    repo_root = _copy_repo_fixture(tmp_path)
    runtime = ShardonRuntime(repo_root=repo_root, gpu_provider=MockGPUProvider())

    with pytest.raises(RuntimeOperationError) as exc:
        runtime._resolve_model_alias(
            model_name="currently-loaded",
            task="chat",
            required_capability="text",
            snapshot=runtime.snapshot(),
        )

    assert exc.value.status_code == 404
    assert exc.value.detail["error"] == "no currently loaded deployment"

