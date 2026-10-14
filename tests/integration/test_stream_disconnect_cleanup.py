from __future__ import annotations

import asyncio
import json

import anyio
import pytest
from shardon_core.api.schemas import ResponseCreateRequest
from shardon_core.auth.service import AuthResult
from shardon_core.gpu.provider import MockGPUProvider
from shardon_core.scheduler.engine import SchedulingDecision
from shardon_core.services.runtime import ShardonRuntime
from shardon_router_api.main import RequestStreamingResponse
from starlette.requests import ClientDisconnect
from test_api_surfaces import _make_repo_copy


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    result = ShardonRuntime(repo_root=_make_repo_copy(tmp_path), gpu_provider=MockGPUProvider())
    result.state_store.mutate(result._seed_snapshot)
    deployment = result.config.deployments["chat-a"]

    def loaded(state):
        state.deployments[deployment.id].loaded = True
        state.deployments[deployment.id].state = "ready"

    result.state_store.mutate(loaded)
    monkeypatch.setattr(result, "refresh_gpu_observations", lambda: None)
    monkeypatch.setattr(result, "enforce_keep_free", lambda: None)
    monkeypatch.setattr(result, "_reconcile_loaded_processes", lambda s: s)
    monkeypatch.setattr(
        result.scheduler,
        "schedule",
        lambda *a: SchedulingDecision(
            True,
            deployment.id,
            deployment.backend_runtime_id,
            deployment.preferred_gpu_group_id(),
            200,
            "ready",
        ),
    )
    yield result
    result.stop_router()


def records(runtime):
    return [
        json.loads(line)
        for p in runtime.usage_store.root.glob("*.jsonl")
        for line in p.read_text().splitlines()
    ]


@pytest.mark.parametrize("failure_at", ["headers", "body", "receive", "cancel"])
def test_disconnect_closes_prefetched_runtime_and_backend(runtime, monkeypatch, failure_at):
    closed = []

    class Adapter:
        async def stream_response(self, payload):
            try:
                yield b'data: {"usage":{"total_tokens":3}}\n\n'
                await asyncio.Event().wait()
            finally:
                # Cleanup must also survive an enclosing AnyIO cancellation scope.
                await anyio.sleep(0)
                closed.append(True)

    monkeypatch.setattr(runtime.backends, "adapter_for", lambda *a, **k: Adapter())

    async def run():
        iterator = runtime.stream_response(
            ResponseCreateRequest(model="demo-chat", input="test", stream=True),
            AuthResult(id="key-a", user_name="alice", priority=100, permissions=["inference"]),
        )
        response = await RequestStreamingResponse.prefetch(iterator)
        assert runtime.snapshot().active_requests
        sent_body = asyncio.Event()

        async def send(message):
            if failure_at == "headers" and message["type"] == "http.response.start":
                raise OSError("client disconnected before headers")
            if message["type"] == "http.response.body":
                sent_body.set()
                if failure_at == "body":
                    raise OSError("client disconnected during send")
                # Suspend sending with the generator paused at yield.
                await asyncio.Event().wait()

        async def receive():
            await sent_body.wait()
            return {"type": "http.disconnect"}

        scope = {
            "type": "http",
            "asgi": {"spec_version": "2.0" if failure_at == "receive" else "2.4"},
        }
        if failure_at == "cancel":
            with anyio.CancelScope() as cancel_scope:

                async def cancel():
                    await sent_body.wait()
                    cancel_scope.cancel()

                async with anyio.create_task_group() as group:
                    group.start_soon(cancel)
                    await response(scope, receive, send)
        elif failure_at == "receive":
            await response(scope, receive, send)
        else:
            with pytest.raises(ClientDisconnect):
                await response(scope, receive, send)
        assert closed == [True]
        snapshot = runtime.snapshot()
        assert not snapshot.active_requests and not snapshot.queued_requests
        assert all(not d.active_request_ids for d in snapshot.deployments.values())
        saved = records(runtime)
        assert len(saved) == 1 and saved[0]["outcome"] == "cancelled"
        assert saved[0]["total_tokens"] == 3
        assert not runtime._request_usage

    asyncio.run(run())
