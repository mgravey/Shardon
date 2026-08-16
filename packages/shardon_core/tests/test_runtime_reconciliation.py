from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import time
from pathlib import Path

import httpx
import pytest

from shardon_core.api.schemas import ChatCompletionRequest, ResponseCreateRequest
from shardon_core.auth.service import AuthResult
from shardon_core.gpu.provider import MockGPUProvider
from shardon_core.services.runtime import RuntimeOperationError, ShardonRuntime
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


def _auth() -> AuthResult:
    return AuthResult(
        id="key-1",
        user_name="alice",
        priority=100,
        permissions=["inference"],
    )


def _chat_request() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="demo-chat",
        messages=[{"role": "user", "content": "ping"}],
    )


def _loaded_chat_state(runtime: ShardonRuntime, *, pid: int = 4_201) -> DeploymentRuntimeState:
    deployment = runtime.config.deployments["chat-a"]
    gpu_group_id = deployment.preferred_gpu_group_id()
    return DeploymentRuntimeState(
        deployment_id=deployment.id,
        gpu_group_id=gpu_group_id,
        selected_gpu_group_id=gpu_group_id,
        backend_runtime_id=deployment.backend_runtime_id,
        loaded=True,
        state="ready",
        desired_state="loaded",
        process_id=pid,
        resident_memory_fraction=deployment.memory_fraction_for_group(gpu_group_id),
    )


def _seed_loaded_chat(runtime: ShardonRuntime, *, pid: int = 4_201) -> None:
    runtime.state_store.mutate(
        lambda snapshot: snapshot.model_copy(
            update={"deployments": {"chat-a": _loaded_chat_state(runtime, pid=pid)}}
        )
    )


def _install_fake_start(
    runtime: ShardonRuntime,
    start_calls: list[str],
    *,
    delay: float = 0,
) -> None:
    async def fake_start_and_ready(deployment, *, gpu_group_id: str):  # type: ignore[no-untyped-def]
        start_calls.append(deployment.id)
        if delay:
            await asyncio.sleep(delay)
        return {
            "deployment_id": deployment.id,
            "backend_runtime_id": deployment.backend_runtime_id,
            "gpu_group_id": gpu_group_id,
            "pid": 5_000 + len(start_calls),
            "attempt_count": 1,
            "ready_at": "2026-08-16T00:00:00+00:00",
            "payload": {"status": "ok"},
        }

    runtime.backends.ensure_started_and_ready = fake_start_and_ready  # type: ignore[method-assign]


class _SuccessfulAdapter:
    async def invoke_chat(self, payload):  # type: ignore[no-untyped-def]
        _ = payload
        return {"id": "response-ok"}


def _mock_stop(runtime: ShardonRuntime, stop_calls: list[tuple[str, bool]]) -> None:
    def stop(deployment_id: str, *, gpu_group_id=None, force: bool = False):  # type: ignore[no-untyped-def]
        _ = gpu_group_id
        stop_calls.append((deployment_id, force))
        runtime.backends.supervisor.processes.pop(deployment_id, None)

    runtime.backends.stop = stop  # type: ignore[method-assign]


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

        async def failing_health(backend_runtime_id: str, *, gpu_group_id: str | None = None) -> dict[str, str]:
            _ = gpu_group_id
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


def test_dead_tracked_pid_relaunches_on_next_request(tmp_path: Path, monkeypatch) -> None:
    runtime = ShardonRuntime(
        repo_root=_copy_repo_fixture(tmp_path),
        gpu_provider=MockGPUProvider(),
    )
    _seed_loaded_chat(runtime, pid=4_301)
    start_calls: list[str] = []
    _install_fake_start(runtime, start_calls)
    runtime.backends.adapter_for = (  # type: ignore[method-assign]
        lambda backend_runtime_id, gpu_group_id=None: _SuccessfulAdapter()
    )

    def process_is_dead(pid: int, signal_number: int) -> None:
        _ = signal_number
        if pid == 4_301:
            raise ProcessLookupError(pid)

    monkeypatch.setattr("shardon_core.services.runtime.os.kill", process_is_dead)

    response = asyncio.run(runtime.route_chat(_chat_request(), _auth()))

    state = runtime.snapshot().deployments["chat-a"]
    assert response == {"id": "response-ok"}
    assert start_calls == ["chat-a"]
    assert state.loaded is True
    assert state.process_id == 5_001


def test_unreachable_health_cleans_up_live_parent_and_next_request_relaunches(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runtime = ShardonRuntime(
        repo_root=_copy_repo_fixture(tmp_path),
        gpu_provider=MockGPUProvider(),
    )
    _seed_loaded_chat(runtime, pid=4_302)
    monkeypatch.setattr("shardon_core.services.runtime.os.kill", lambda pid, signal_number: None)
    stop_calls: list[tuple[str, bool]] = []
    start_calls: list[str] = []
    _mock_stop(runtime, stop_calls)
    _install_fake_start(runtime, start_calls)

    async def failing_health(backend_runtime_id: str, *, gpu_group_id=None):  # type: ignore[no-untyped-def]
        _ = backend_runtime_id
        _ = gpu_group_id
        raise httpx.ConnectError("backend refused connection")

    runtime.backends.health = failing_health  # type: ignore[method-assign]
    asyncio.run(runtime.refresh_backend_health())

    unavailable = runtime.snapshot().deployments["chat-a"]
    assert unavailable.loaded is False
    assert unavailable.state == "failed"
    assert unavailable.process_id is None
    assert stop_calls == [("chat-a", False)]
    events = [
        json.loads(line)
        for line in (runtime.state_root / "events" / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    recovery_event = next(
        event
        for event in events
        if event["category"] == "backend.recovery"
        and event["data"].get("recovery_result") == "marked_unavailable"
    )
    assert recovery_event["data"]["deployment_id"] == "chat-a"
    assert recovery_event["data"]["gpu_group_id"] == "group-a"
    assert recovery_event["data"]["old_pid"] == 4_302
    assert "health unreachable" in recovery_event["data"]["reason"]

    runtime.backends.adapter_for = (  # type: ignore[method-assign]
        lambda backend_runtime_id, gpu_group_id=None: _SuccessfulAdapter()
    )
    response = asyncio.run(runtime.route_chat(_chat_request(), _auth()))
    assert response == {"id": "response-ok"}
    assert start_calls == ["chat-a"]


def test_request_connection_failure_recovers_at_most_once(tmp_path: Path, monkeypatch) -> None:
    runtime = ShardonRuntime(
        repo_root=_copy_repo_fixture(tmp_path),
        gpu_provider=MockGPUProvider(),
    )
    _seed_loaded_chat(runtime, pid=4_303)
    monkeypatch.setattr("shardon_core.services.runtime.os.kill", lambda pid, signal_number: None)
    stop_calls: list[tuple[str, bool]] = []
    start_calls: list[str] = []
    invocation_count = 0
    _mock_stop(runtime, stop_calls)
    _install_fake_start(runtime, start_calls)

    class FailingAdapter:
        async def invoke_chat(self, payload):  # type: ignore[no-untyped-def]
            nonlocal invocation_count
            _ = payload
            invocation_count += 1
            request = httpx.Request("POST", "http://backend.test/v1/chat/completions")
            raise httpx.ConnectError("backend disconnected", request=request)

    runtime.backends.adapter_for = (  # type: ignore[method-assign]
        lambda backend_runtime_id, gpu_group_id=None: FailingAdapter()
    )

    with pytest.raises(RuntimeOperationError) as exc_info:
        asyncio.run(runtime.route_chat(_chat_request(), _auth()))

    assert invocation_count == 2
    assert start_calls == ["chat-a"]
    assert stop_calls == [("chat-a", False)]
    assert exc_info.value.detail["recovery_attempts"] == 1
    events = [
        json.loads(line)
        for line in (runtime.state_root / "events" / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    succeeded = next(
        event for event in events if event["category"] == "backend.recovery_succeeded"
    )
    assert succeeded["data"]["old_pid"] == 4_303
    assert succeeded["data"]["new_pid"] == 5_001
    assert succeeded["data"]["recovery_result"] == "ready"


def test_concurrent_requests_start_backend_once(tmp_path: Path) -> None:
    runtime = ShardonRuntime(
        repo_root=_copy_repo_fixture(tmp_path),
        gpu_provider=MockGPUProvider(),
    )
    start_calls: list[str] = []
    _install_fake_start(runtime, start_calls, delay=0.05)
    runtime.backends.adapter_for = (  # type: ignore[method-assign]
        lambda backend_runtime_id, gpu_group_id=None: _SuccessfulAdapter()
    )

    async def run_concurrently():
        return await asyncio.gather(
            runtime.route_chat(_chat_request(), _auth()),
            runtime.route_chat(_chat_request(), _auth()),
        )

    responses = asyncio.run(run_concurrently())

    assert responses == [{"id": "response-ok"}, {"id": "response-ok"}]
    assert start_calls == ["chat-a"]


def test_stream_connection_failure_recovers_before_first_chunk(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runtime = ShardonRuntime(
        repo_root=_copy_repo_fixture(tmp_path),
        gpu_provider=MockGPUProvider(),
    )
    _seed_loaded_chat(runtime, pid=4_304)
    monkeypatch.setattr("shardon_core.services.runtime.os.kill", lambda pid, signal_number: None)
    stop_calls: list[tuple[str, bool]] = []
    start_calls: list[str] = []
    stream_attempts = 0
    _mock_stop(runtime, stop_calls)
    _install_fake_start(runtime, start_calls)

    class StreamingAdapter:
        async def stream_response(self, payload):  # type: ignore[no-untyped-def]
            nonlocal stream_attempts
            _ = payload
            stream_attempts += 1
            if stream_attempts == 1:
                request = httpx.Request("POST", "http://backend.test/v1/responses")
                raise httpx.ConnectError("stream could not connect", request=request)
            yield b"data: recovered\n\n"

    runtime.backends.adapter_for = (  # type: ignore[method-assign]
        lambda backend_runtime_id, gpu_group_id=None: StreamingAdapter()
    )

    async def collect_stream() -> list[bytes]:
        request = ResponseCreateRequest(model="demo-chat", input="ping", stream=True)
        return [chunk async for chunk in runtime.stream_response(request, _auth())]

    chunks = asyncio.run(collect_stream())

    assert chunks == [b"data: recovered\n\n"]
    assert stream_attempts == 2
    assert start_calls == ["chat-a"]
    assert stop_calls == [("chat-a", False)]


def test_backend_http_4xx_does_not_restart(tmp_path: Path, monkeypatch) -> None:
    runtime = ShardonRuntime(
        repo_root=_copy_repo_fixture(tmp_path),
        gpu_provider=MockGPUProvider(),
    )
    _seed_loaded_chat(runtime, pid=4_305)
    monkeypatch.setattr("shardon_core.services.runtime.os.kill", lambda pid, signal_number: None)
    stop_calls: list[tuple[str, bool]] = []
    start_calls: list[str] = []
    _mock_stop(runtime, stop_calls)
    _install_fake_start(runtime, start_calls)

    class InvalidRequestAdapter:
        async def invoke_chat(self, payload):  # type: ignore[no-untyped-def]
            _ = payload
            request = httpx.Request("POST", "http://backend.test/v1/chat/completions")
            response = httpx.Response(400, request=request, json={"error": "invalid payload"})
            raise httpx.HTTPStatusError("invalid payload", request=request, response=response)

    runtime.backends.adapter_for = (  # type: ignore[method-assign]
        lambda backend_runtime_id, gpu_group_id=None: InvalidRequestAdapter()
    )

    with pytest.raises(RuntimeOperationError) as exc_info:
        asyncio.run(runtime.route_chat(_chat_request(), _auth()))

    assert exc_info.value.detail["recovery_attempts"] == 0
    assert start_calls == []
    assert stop_calls == []


def test_manual_unload_remains_eligible_for_on_demand_load(tmp_path: Path) -> None:
    runtime = ShardonRuntime(
        repo_root=_copy_repo_fixture(tmp_path),
        gpu_provider=MockGPUProvider(),
    )
    stop_calls: list[tuple[str, bool]] = []
    start_calls: list[str] = []
    _mock_stop(runtime, stop_calls)
    _install_fake_start(runtime, start_calls)
    runtime.backends.adapter_for = (  # type: ignore[method-assign]
        lambda backend_runtime_id, gpu_group_id=None: _SuccessfulAdapter()
    )

    asyncio.run(runtime.load_deployment(deployment_id="chat-a", actor="test"))
    asyncio.run(runtime.unload_deployment("chat-a", actor="test"))
    response = asyncio.run(runtime.route_chat(_chat_request(), _auth()))

    assert response == {"id": "response-ok"}
    assert start_calls == ["chat-a", "chat-a"]
    assert stop_calls == [("chat-a", False)]


def test_repeated_background_failures_do_not_escape_scheduler_tick(tmp_path: Path) -> None:
    from shardon_router_api.main import _run_background_tick

    runtime = ShardonRuntime(
        repo_root=_copy_repo_fixture(tmp_path),
        gpu_provider=MockGPUProvider(),
    )
    health_attempts = 0

    async def failing_refresh():
        nonlocal health_attempts
        health_attempts += 1
        raise RuntimeError("health loop exploded")

    runtime.refresh_backend_health = failing_refresh  # type: ignore[method-assign]

    async def run_ticks() -> None:
        await _run_background_tick(runtime)
        await _run_background_tick(runtime)

    asyncio.run(run_ticks())

    assert health_attempts == 2
    events = [
        json.loads(line)
        for line in (runtime.state_root / "events" / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    background_failures = [
        event for event in events if event["category"] == "router.background_failed"
    ]
    assert len(background_failures) == 2
