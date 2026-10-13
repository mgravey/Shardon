from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import pytest
from shardon_core.api.schemas import (
    BatchCreateRequest,
    ChatCompletionRequest,
    ResponseCreateRequest,
)
from shardon_core.auth.service import AuthResult
from shardon_core.gpu.provider import MockGPUProvider
from shardon_core.scheduler.engine import SchedulingDecision
from shardon_core.services.observability import (
    StreamUsage,
    TokenUsage,
    UsageStore,
    activity_snapshot,
    extract_usage,
)
from shardon_core.services.runtime import RuntimeOperationError, ShardonRuntime
from shardon_core.state.models import ActiveRequest, BatchJobState, RuntimeStateSnapshot
from test_runtime_queue import _copy_repo_fixture, _loaded_deployment_state

NOW = datetime(2026, 10, 13, 12, tzinfo=UTC)


def request(request_id="r1", **changes):
    data = dict(
        id=request_id,
        api_key_id="key-a",
        user_name="alice",
        model_name="demo-chat",
        deployment_id="chat-a",
        backend_runtime_id="mock",
        gpu_group_id="group-a",
        request_class="interactive",
        status="queued",
        priority=100,
        created_at="2026-10-13T10:00:00+00:00",
        task="chat",
    )
    return ActiveRequest(**(data | changes))


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    result = ShardonRuntime(repo_root=_copy_repo_fixture(tmp_path), gpu_provider=MockGPUProvider())
    result.state_store.mutate(
        lambda state: state.model_copy(
            update={
                "deployments": {"chat-a": _loaded_deployment_state(result, "chat-a")},
            }
        )
    )
    monkeypatch.setattr(result, "refresh_gpu_observations", lambda: None)
    monkeypatch.setattr(result, "enforce_keep_free", lambda: None)
    monkeypatch.setattr(result, "_reconcile_loaded_processes", lambda state: state)
    monkeypatch.setattr(
        result.scheduler,
        "schedule",
        lambda *args: SchedulingDecision(
            True,
            "chat-a",
            result.config.deployments["chat-a"].backend_runtime_id,
            result.config.deployments["chat-a"].preferred_gpu_group_id(),
            200,
            "ready",
        ),
    )
    return result


def auth():
    return AuthResult(id="key-a", user_name="alice", priority=100, permissions=["inference"])


def chat():
    return ChatCompletionRequest(
        model="demo-chat", messages=[{"role": "user", "content": "secret"}]
    )


def records(runtime):
    return [
        json.loads(line)
        for p in (runtime.state_root / "usage").glob("*.jsonl")
        for line in p.read_text().splitlines()
    ]


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"usage": {"prompt_tokens": 12, "completion_tokens": 3}}, (12, 3, 15)),
        (
            {"response": {"usage": {"input_tokens": 4, "output_tokens": 2, "total_tokens": 6}}},
            (4, 2, 6),
        ),
        ({"usage": {"prompt_tokens": 5, "total_tokens": 5}}, (5, None, 5)),
        ({"usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}}, (0, 0, 0)),
        (
            {"usage": {"total_tokens": True, "input_tokens": -1, "output_tokens": "2"}},
            (None, None, None),
        ),
        ({"text": "no accounting"}, (None, None, None)),
    ],
)
def test_usage_normalization(payload, expected):
    assert tuple(extract_usage(payload).model_dump().values()) == expected


@pytest.mark.parametrize("chunk_size", [1, 2, 7, 4096])
def test_stream_fragmentation_and_cumulative_usage(chunk_size):
    wire = (
        'data: {"choices":[{"text":"café"}]}\r\n\r\n'
        'data: {"usage":{"input_tokens":4,"output_tokens":1,"total_tokens":5}}\r\n\r\n'
        'data: {"response":{"usage":{"input_tokens":4,"output_tokens":3,"total_tokens":7}}}\n\n'
        'data: {"usage":{"total_tokens":7}}\n\ndata: [DONE]\n\n'
    ).encode()
    observer = StreamUsage()
    for i in range(0, len(wire), chunk_size):
        observer.feed(wire[i : i + chunk_size])
    assert observer.finish() == TokenUsage(input_tokens=4, output_tokens=3, total_tokens=7)


def test_oversize_and_malformed_sse_recovers():
    observer = StreamUsage()
    observer.feed(b"data: " + b"x" * (1024 * 1024 + 20) + b"\n\n")
    observer.feed(b'data: not json\n\ndata: {"usage":{"total_tokens":9}}\n\n')
    assert observer.finish().total_tokens == 9


def test_activity_deduplicates_and_filters(runtime):
    active = request(status="running", started_at="2026-10-13T11:00:00+00:00")
    loading = request("r2", phase="loading", wait_reason="Loading model")
    waiting = request("r3", user_name="bob", api_key_id="key-b", model_name="second")
    state = RuntimeStateSnapshot(
        queued_requests=[active, loading, loading, waiting], active_requests={active.id: active}
    )
    result = activity_snapshot(state, runtime.config, now=NOW)
    assert (result.in_flight, result.waiting, result.loading) == (1, 2, 1)
    row = next(r for r in result.models if r.model_name == "demo-chat")
    assert row.oldest_wait_seconds == 7200
    assert row.requests[0].phase == "in_flight"
    filtered = activity_snapshot(state, runtime.config, api_key="key-b", user="bob", now=NOW)
    assert [r.model_name for r in filtered.models] == ["second"]
    assert filtered.waiting == 1


def test_batch_projection_does_not_double_count_current_item(runtime):
    state = RuntimeStateSnapshot(
        batch_jobs={
            "b1": BatchJobState(
                id="b1",
                api_key_id="key-a",
                user_name="alice",
                model_name="demo-chat",
                status="running",
                created_at=NOW.isoformat(),
                updated_at=NOW.isoformat(),
                total_items=6,
                completed_items=2,
                failed_items=1,
                current_item=4,
            )
        }
    )
    row = activity_snapshot(state, runtime.config, now=NOW).batches[0]
    assert row.current_item == 4
    assert row.remaining_items == 2


def test_storage_concurrency_restart_partial_tail_and_dedup(tmp_path):
    def write(index):
        store = UsageStore(tmp_path)
        item = request(f"r{index % 12}", finished_at=NOW.isoformat())
        store.record(item, "completed", TokenUsage(input_tokens=2, output_tokens=3, total_tokens=5))

    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(write, range(48)))
    path = tmp_path / "usage" / "2026-10-13.jsonl"
    with path.open("a") as handle:
        handle.write('{"truncated":')
    store = UsageStore(tmp_path)
    store.record(request("last", finished_at=NOW.isoformat()), "failed")
    result = UsageStore(tmp_path).summary(now=NOW)
    assert result.totals.requests == 13
    assert result.totals.total_tokens == 60
    assert result.totals.failed == 1
    assert result.totals.usage_available == 12
    assert "secret" not in path.read_text()


def test_windows_filters_unknowns_and_cache(tmp_path, monkeypatch):
    store = UsageStore(tmp_path)
    for index, date in enumerate(
        ["2026-09-13", "2026-09-14", "2026-10-06", "2026-10-07", "2026-10-13"]
    ):
        store.record(
            request(f"r{index}", finished_at=f"{date}T00:00:00+00:00"),
            "completed",
            TokenUsage(total_tokens=10) if index != 4 else None,
        )
    assert store.summary(period="30d", now=NOW).totals.requests == 4
    assert store.summary(period="7d", now=NOW).totals.requests == 2
    today = store.summary(now=NOW)
    assert today.totals.requests == 1
    assert today.totals.total_tokens is None
    assert today.totals.usage_coverage == 0
    monkeypatch.setattr(store, "_read", lambda path: pytest.fail("aggregate was not cached"))
    assert store.summary(now=NOW).totals.requests == 1
    monkeypatch.undo()
    store.record(
        request("extra", user_name="alice", api_key_id="key-b", finished_at=NOW.isoformat()),
        "completed",
        TokenUsage(total_tokens=20),
    )
    assert store.summary(group_by="user", now=NOW).rows[0].requests == 2
    assert (
        store.summary(api_key="key-b", model="demo-chat", user="alice", now=NOW).totals.total_tokens
        == 20
    )
    assert store.summary(model="missing", now=NOW).rows == []


def test_old_state_defaults():
    old = request().model_dump(
        exclude={"task", "requested_model_name", "phase", "wait_reason", "batch_id"}
    )
    parsed = RuntimeStateSnapshot.model_validate({"queued_requests": [old]})
    assert parsed.queued_requests[0].phase == "waiting"
    assert parsed.queued_requests[0].task is None


def test_route_success_and_retry_one_record(runtime, monkeypatch):
    calls = 0

    class Adapter:
        async def invoke_chat(self, payload):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise ConnectionError("temporary")
            return {"usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}}

    async def recover(*args, **kwargs):
        pass

    monkeypatch.setattr(runtime.backends, "adapter_for", lambda *a, **k: Adapter())
    monkeypatch.setattr(runtime, "_recover_backend_for_request", recover)
    asyncio.run(runtime.route_chat(chat(), auth()))
    saved = records(runtime)
    assert calls == 2 and len(saved) == 1
    assert saved[0]["outcome"] == "completed" and saved[0]["total_tokens"] == 12
    assert saved[0]["task"] == "chat" and saved[0]["requested_model_name"] == "demo-chat"
    assert runtime.state_store.load().active_requests == {}


@pytest.mark.parametrize("stage", ["waiting", "loading", "in_flight"])
def test_cancellation_cleans_all_references(runtime, monkeypatch, stage):
    async def run():
        entered = asyncio.Event()

        async def blocked(*args, **kwargs):
            entered.set()
            await asyncio.Event().wait()

        class Adapter:
            invoke_chat = blocked

        monkeypatch.setattr(runtime.backends, "adapter_for", lambda *a, **k: Adapter())
        if stage == "waiting":

            def schedule(*args):
                entered.set()
                return SchedulingDecision(False, None, None, None, 409, "GPU is busy")

            monkeypatch.setattr(runtime.scheduler, "schedule", schedule)
        elif stage == "loading":
            runtime.state_store.mutate(lambda s: s.model_copy(update={"deployments": {}}))
            monkeypatch.setattr(runtime, "_start_and_mark_ready", blocked)
        task = asyncio.create_task(runtime.route_chat(chat(), auth()))
        await entered.wait()
        snapshot = runtime.state_store.load()
        row = next(
            r
            for r in activity_snapshot(snapshot, runtime.config).models
            if r.model_name == "demo-chat"
        )
        assert row.requests[0].phase == stage
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    state = runtime.state_store.load()
    assert not state.active_requests and not state.queued_requests
    assert all(not d.active_request_ids for d in state.deployments.values())
    assert records(runtime)[0]["outcome"] == "cancelled"


def test_timeout_and_clear_queue_preserves_in_flight(runtime, monkeypatch):
    runtime.config.global_config.interactive_request_timeout_seconds = 0
    with pytest.raises(RuntimeOperationError):
        asyncio.run(runtime.route_chat(chat(), auth()))
    assert records(runtime)[0]["outcome"] == "failed"
    active = request(status="running")
    waiting = request("wait")
    runtime.state_store.mutate(
        lambda s: s.model_copy(
            update={
                "queued_requests": [active, waiting],
                "active_requests": {active.id: active},
            }
        )
    )
    result = runtime.clear_queue(actor="test")
    assert result["interactive_request_ids"] == ["wait"]
    assert runtime.state_store.load().active_requests[active.id].status == "running"
    assert next(r for r in records(runtime) if r["request_id"] == "wait")["outcome"] == "cancelled"


def test_observation_unchanged_does_not_write(runtime, monkeypatch):
    runtime.state_store.mutate(lambda s: s.model_copy(update={"queued_requests": [request()]}))
    runtime._update_request_observation("r1", wait_reason="busy")
    monkeypatch.setattr(runtime.state_store, "mutate", lambda fn: pytest.fail("redundant write"))
    runtime._update_request_observation("r1", wait_reason="busy")


def test_stream_usage_forwarding_and_close(runtime, monkeypatch):
    wire = b'data: {"response":{"usage":{"input_tokens":2,"output_tokens":3,"total_tokens":5}}}\n\n'

    class Adapter:
        async def stream_response(self, payload):
            for chunk in [wire[:12], wire[12:]]:
                yield chunk

    monkeypatch.setattr(runtime.backends, "adapter_for", lambda *a, **k: Adapter())

    async def run():
        stream = runtime.stream_response(
            ResponseCreateRequest(model="demo-chat", input="secret", stream=True), auth()
        )
        assert b"".join([chunk async for chunk in stream]) == wire

    asyncio.run(run())
    assert records(runtime)[0]["total_tokens"] == 5
    assert len(records(runtime)) == 1


def test_batch_items_have_individual_usage(runtime, monkeypatch):
    calls = 0

    class Adapter:
        async def invoke_chat(self, payload):
            nonlocal calls
            calls += 1
            state = runtime.state_store.load()
            batch = activity_snapshot(state, runtime.config).batches[0]
            assert batch.current_item == calls
            assert batch.remaining_items == 3 - calls
            if calls == 2:
                raise ValueError("failed item")
            return {"usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}}

    monkeypatch.setattr(runtime.backends, "adapter_for", lambda *a, **k: Adapter())

    async def run():
        await runtime.submit_batch(
            BatchCreateRequest(model="demo-chat", requests=[{}, {}, {}]), auth()
        )
        await runtime.process_one_batch()

    asyncio.run(run())
    saved = records(runtime)
    assert len(saved) == 3
    assert sum(r["total_tokens"] or 0 for r in saved) == 10
    assert [r["outcome"] for r in saved] == ["completed", "failed", "completed"]
    assert all(r["batch_id"] and r["api_key_id"] == "key-a" for r in saved)
    assert not runtime.state_store.load().active_requests


def test_closing_stream_immediately_releases_request_and_backend(runtime, monkeypatch):
    closed = []

    class Adapter:
        async def stream_response(self, payload):
            try:
                yield b'data: {"usage":{"total_tokens":3}}\n\n'
                await asyncio.Event().wait()
            finally:
                closed.append(True)

    monkeypatch.setattr(runtime.backends, "adapter_for", lambda *a, **k: Adapter())

    async def run():
        stream = runtime.stream_response(
            ResponseCreateRequest(model="demo-chat", input="secret", stream=True), auth()
        )
        await anext(stream)
        assert runtime.state_store.load().active_requests
        await stream.aclose()
        assert not runtime.state_store.load().active_requests
        assert not runtime.state_store.load().queued_requests
        assert closed == [True]
        saved = records(runtime)
        assert len(saved) == 1 and saved[0]["outcome"] == "cancelled"
        assert saved[0]["total_tokens"] == 3

    asyncio.run(run())


def test_batch_cancellation_accounts_pending_items(runtime, monkeypatch):
    async def run():
        entered = asyncio.Event()

        class Adapter:
            async def invoke_chat(self, payload):
                entered.set()
                await asyncio.Event().wait()

        monkeypatch.setattr(runtime.backends, "adapter_for", lambda *a, **k: Adapter())
        await runtime.submit_batch(
            BatchCreateRequest(model="demo-chat", requests=[{}, {}, {}]), auth()
        )
        task = asyncio.create_task(runtime.process_one_batch())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        state = runtime.state_store.load()
        job = next(iter(state.batch_jobs.values()))
        assert job.status == "cancelled" and job.cancelled_items == 3
        assert not state.active_requests and not state.deployments["chat-a"].active_request_ids
        assert len(records(runtime)) == 3

    asyncio.run(run())


def test_duplicate_finalization_across_midnight(tmp_path):
    item = request(
        "midnight", created_at="2026-10-12T23:59:00+00:00", finished_at="2026-10-12T23:59:59+00:00"
    )
    UsageStore(tmp_path).record(item, "completed", TokenUsage(total_tokens=5))
    item.finished_at = "2026-10-13T00:00:01+00:00"
    store = UsageStore(tmp_path)
    store.record(item, "completed", TokenUsage(total_tokens=5))
    assert store.summary(period="7d", now=NOW).totals.requests == 1
    assert store.summary(period="today", now=NOW).totals.requests == 0


@pytest.mark.parametrize("cancel", [False, True])
def test_batch_load_failure_or_clear_does_not_leave_queued_work(runtime, monkeypatch, cancel):
    group = runtime.config.deployments["chat-a"].preferred_gpu_group_id()
    monkeypatch.setattr(
        runtime.scheduler,
        "schedule",
        lambda *a: SchedulingDecision(
            True,
            "chat-a",
            "mock",
            group,
            200,
            "load",
            should_load=True,
        ),
    )

    async def start(*a, **k):
        if cancel:
            runtime.clear_queue(clear_batches=True, actor="test")
        else:
            raise RuntimeOperationError("load failed", detail={"error": "failed"})

    monkeypatch.setattr(runtime, "_start_and_mark_ready", start)

    async def run():
        await runtime.submit_batch(BatchCreateRequest(model="demo-chat", requests=[{}, {}]), auth())
        if cancel:
            await runtime.process_one_batch()
        else:
            with pytest.raises(RuntimeOperationError):
                await runtime.process_one_batch()

    asyncio.run(run())
    job = next(iter(runtime.state_store.load().batch_jobs.values()))
    assert job.status == ("cancelled" if cancel else "failed")
    saved = records(runtime)
    assert len(saved) == 2 and all(r["deployment_id"] == "chat-a" for r in saved)


def test_stream_failure_event_preserves_bytes_and_accounts_failure(runtime, monkeypatch):
    wire = b'data: {"type":"response.failed","response":{"usage":{"total_tokens":4}}}\n\n'

    class Adapter:
        async def stream_response(self, payload):
            yield wire

    monkeypatch.setattr(runtime.backends, "adapter_for", lambda *a, **k: Adapter())

    async def run():
        stream = runtime.stream_response(
            ResponseCreateRequest(model="demo-chat", input="x", stream=True), auth()
        )
        assert b"".join([chunk async for chunk in stream]) == wire

    asyncio.run(run())
    saved = records(runtime)
    assert len(saved) == 1 and saved[0]["outcome"] == "failed"
    assert saved[0]["total_tokens"] == 4
    assert not runtime.state_store.load().active_requests


def test_usage_model_view_groups_requested_alias_and_filters(tmp_path):
    store = UsageStore(tmp_path)
    for index, (actual, requested) in enumerate(
        [
            ("model-a", "currently-loaded"),
            ("model-b", "currently-loaded"),
            ("model-a", "model-a"),
            ("model-b", None),
        ]
    ):
        store.record(
            request(
                f"view-{index}",
                model_name=actual,
                requested_model_name=requested,
                finished_at=NOW.isoformat(),
            ),
            "completed",
            TokenUsage(total_tokens=10),
        )
    actual = store.summary(now=NOW)
    requested = store.summary(now=NOW, model_view="requested")
    assert actual.model_view == "actual" and requested.model_view == "requested"
    assert actual.totals == requested.totals
    assert {m.model_name: m.total_tokens for m in actual.rows[0].models} == {
        "model-a": 20,
        "model-b": 20,
    }
    assert {m.model_name: m.total_tokens for m in requested.rows[0].models} == {
        "currently-loaded": 20,
        "model-a": 10,
        "model-b": 10,
    }
    assert (
        store.summary(now=NOW, model_view="requested", model="currently-loaded").totals.requests
        == 2
    )
    assert store.summary(now=NOW, model="currently-loaded").totals.requests == 0
    assert store.summary(now=NOW).model_dump() == actual.model_dump()
    assert store.summary(now=NOW, model_view="requested").model_dump() == requested.model_dump()


def test_currently_loaded_request_records_both_names(runtime, monkeypatch):
    class Adapter:
        async def invoke_chat(self, payload):
            assert payload["model"] == "demo-chat"
            return {"usage": {"total_tokens": 8}}

    monkeypatch.setattr(runtime.backends, "adapter_for", lambda *a, **k: Adapter())
    payload = chat().model_copy(update={"model": "currently-loaded"})
    asyncio.run(runtime.route_chat(payload, auth()))
    saved = records(runtime)
    assert saved[0]["model_name"] == "demo-chat"
    assert saved[0]["requested_model_name"] == "currently-loaded"
    assert (
        runtime.usage_store.summary(model_view="requested").rows[0].models[0].model_name
        == "currently-loaded"
    )
