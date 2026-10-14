from __future__ import annotations

import json
import select
import subprocess
import sys
import uuid
from contextlib import contextmanager

import pytest
from shardon_core.gpu.provider import MockGPUProvider
from shardon_core.services.observability import TokenUsage
from shardon_core.services.runtime import ShardonRuntime
from shardon_core.state.models import BatchJobState
from shardon_core.state.ownership import RouterInstance
from test_observability import records, request
from test_runtime_queue import _copy_repo_fixture


@pytest.fixture
def runtime(tmp_path):
    result = ShardonRuntime(repo_root=_copy_repo_fixture(tmp_path), gpu_provider=MockGPUProvider())
    yield result
    result.stop_router()


@contextmanager
def router_process(runtime):
    # A real OS lock, held by another process and released by SIGKILL without
    # Python finally/cancellation handlers. The child persists admitted requests.
    script = """
import sys
from pathlib import Path
from shardon_core.services.runtime import ShardonRuntime
from shardon_core.state.models import ActiveRequest
runtime = ShardonRuntime(repo_root=Path(sys.argv[1]))
runtime.start_router()
def admit(state):
    for phase in ('waiting', 'loading', 'in_flight'):
        item = ActiveRequest(
            id=phase, user_name='alice', api_key_id='key-a', model_name='demo-chat',
            deployment_id='chat-a', backend_runtime_id='mock', gpu_group_id='group-a',
            request_class='interactive', priority=100, created_at='2000-01-01T00:00:00+00:00',
            phase=phase, status='running' if phase == 'in_flight' else 'queued',
        )
        runtime._enqueue_request(state, item)
        if phase == 'in_flight':
            state.active_requests[item.id] = item
            for deployment in state.deployments.values():
                deployment.active_request_ids.append(item.id)
    return state
runtime.state_store.mutate(admit)
print('ready', flush=True)
sys.stdin.read()
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(runtime.repo_root)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert select.select([process.stdout], [], [], 15)[0], "router startup timed out"
        assert process.stdout.readline().strip() == "ready"
        yield process
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=10)


def assert_cleared(runtime):
    state = runtime.snapshot()
    assert not state.active_requests and not state.queued_requests
    assert all(not d.active_request_ids for d in state.deployments.values())
    assert json.loads(runtime.state_store.requests_path.read_text()) == {}
    assert json.loads(runtime.state_store.queue_path.read_text()) == []


def test_sigkill_restart_recovers_all_phases_and_recovery_is_idempotent(runtime):
    with router_process(runtime) as process:
        process.kill()
        process.wait(timeout=10)
        # Building another container (including admin startup) does not recover.
        restarted = ShardonRuntime(repo_root=runtime.repo_root, gpu_provider=MockGPUProvider())
        assert len(restarted.snapshot().queued_requests) == 3
        restarted.recover_interrupted_requests()
        assert not records(restarted)
        try:
            restarted.start_router()
            assert_cleared(restarted)
            saved = records(restarted)
            assert len(saved) == 3
            assert all(r["outcome"] == "failed" for r in saved)
            assert all("terminated" in r["interruption_reason"] for r in saved)
            assert all(r["total_tokens"] is None for r in saved)
            restarted.recover_interrupted_requests()
            restarted.start_router()
            assert records(restarted) == saved
        finally:
            restarted.stop_router()


def test_live_owner_is_preserved_regardless_of_age_and_recovered_after_exit(runtime):
    with router_process(runtime) as process:
        before = runtime.snapshot()
        runtime.start_router()
        runtime.recover_interrupted_requests()
        assert runtime.snapshot() == before
        assert not records(runtime)
        process.kill()
        process.wait(timeout=10)
        # A surviving router can recover a dead peer without restarting itself.
        runtime.recover_interrupted_requests()
        assert_cleared(runtime)
        assert len(records(runtime)) == 3


def test_same_process_instances_and_legacy_unknown_owners_are_preserved(runtime):
    owner = RouterInstance(runtime.state_root)
    items = [
        request("live", router_instance_id=owner.id),
        request("legacy"),
        request("missing", router_instance_id=uuid.uuid4().hex),
        request("invalid", router_instance_id="../runtime"),
    ]
    runtime.state_store.mutate(lambda s: s.queued_requests.extend(items))
    try:
        runtime.start_router()
        assert runtime.snapshot().queued_requests == items
        owner.close()
        runtime.recover_interrupted_requests()
        assert [r.id for r in runtime.snapshot().queued_requests] == [
            "legacy",
            "missing",
            "invalid",
        ]
        assert [r["request_id"] for r in records(runtime)] == ["live"]
    finally:
        owner.close()


def test_recovery_replays_after_usage_commit_without_duplicate_accounting(runtime, monkeypatch):
    from shardon_core.state import store

    with router_process(runtime) as process:
        # Simulate successful accounting committed before the router was killed
        # during state cleanup, including a previous UTC day's usage record.
        item = runtime.snapshot().active_requests["in_flight"]
        item.finished_at = "2026-10-13T23:59:59+00:00"
        runtime.usage_store.record(item, "completed", TokenUsage(total_tokens=7))
        process.kill()
        process.wait(timeout=10)
        original = store.atomic_write_json

        def fail_snapshot(path, payload):
            if path == runtime.state_store.snapshot_path:
                raise OSError("simulated crash before runtime state commit")
            original(path, payload)

        with monkeypatch.context() as patch:
            patch.setattr(store, "atomic_write_json", fail_snapshot)
            with pytest.raises(OSError):
                runtime.start_router()
        assert runtime.snapshot().active_requests
        runtime.start_router()
        runtime.recover_interrupted_requests()
        assert_cleared(runtime)
        saved = records(runtime)
        assert len(saved) == 3
        committed = next(r for r in saved if r["request_id"] == "in_flight")
        assert committed["outcome"] == "completed" and committed["total_tokens"] == 7


@pytest.mark.parametrize("status", ["queued", "running"])
def test_interrupted_batch_accounts_remaining_items_and_preserves_committed_usage(runtime, status):
    owner = RouterInstance(runtime.state_root)
    job = BatchJobState(
        id="batch",
        router_instance_id=owner.id,
        api_key_id="key-a",
        user_name="alice",
        model_name="demo-chat",
        status=status,
        total_items=3,
        completed_items=1,
        created_at="2026-10-13T10:00:00+00:00",
        updated_at="2026-10-13T10:00:00+00:00",
    )
    item = runtime._batch_request(job, 1)
    runtime.usage_store.record(item, "completed", TokenUsage(total_tokens=9))

    def seed(state):
        state.batch_jobs[job.id] = job
        state.active_requests[item.id] = item

    runtime.state_store.mutate(seed)
    owner.close()
    runtime.start_router()
    runtime.recover_interrupted_requests()
    job = runtime.snapshot().batch_jobs[job.id]
    assert job.status == "failed" and job.interruption_reason
    assert (job.completed_items, job.failed_items) == (2, 1)
    assert not runtime.snapshot().active_requests
    assert len(records(runtime)) == 2
    assert sum(r["total_tokens"] or 0 for r in records(runtime)) == 9
