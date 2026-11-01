from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest
from shardon_core.api.schemas import ChatCompletionRequest
from shardon_core.config.schemas import SleepConfig, SleepPolicy
from shardon_core.services.runtime import RuntimeOperationError
from shardon_core.state.models import ActiveRequest, BatchJobState
from shardon_core.utils.time import utc_now, utc_now_iso
from sleep_fixtures import (
    AUTH,
    build_repo,
    deployment,
    events,
    make_runtime,
    mock_backend,
    mock_stats,
    shutdown,
)

SLEEP = {"enabled": True, "idle_seconds": 600, "wake_timeout_seconds": 20}


@pytest.fixture
def runtimes():
    created = []
    yield created
    for runtime in created:
        shutdown(runtime)


def _single(tmp_path, runtimes, *, sleep=None, environment=None, global_overrides=None):
    repo = build_repo(
        tmp_path,
        backends=[
            mock_backend(
                "vllm-a",
                sleep={**SLEEP, **(sleep or {})},
                environment=environment,
            )
        ],
        deployments=[deployment("dep-a", "vllm-a")],
        global_overrides=global_overrides,
    )
    runtime = make_runtime(repo)
    runtimes.append(runtime)
    asyncio.run(runtime.load_deployment(deployment_id="dep-a", actor="test"))
    return runtime


def _chat(model: str = "model-dep-a") -> ChatCompletionRequest:
    return ChatCompletionRequest(model=model, messages=[{"role": "user", "content": "hi"}])


def _age_activity(runtime, deployment_id: str, seconds: int) -> None:
    old = (utc_now() - timedelta(seconds=seconds)).isoformat()

    def mutate(state):
        item = state.deployments[deployment_id]
        item.last_used_at = old
        item.loaded_at = old
        item.readiness_passed_at = old
        item.woke_at = None
        return state

    runtime.state_store.mutate(mutate)


def test_sleep_policy_defaults_and_deployment_override() -> None:
    assert SleepPolicy.resolve(None, None).enabled is False
    policy = SleepPolicy.resolve(
        SleepConfig(enabled=True, level=2, idle_seconds=60),
        SleepConfig(idle_seconds=30, sleep_on_evict=True),
    )
    assert policy.enabled is True
    assert policy.level == 2
    assert policy.idle_seconds == 30
    assert policy.sleep_on_evict is True
    # Ten minutes of idleness before sleeping, unless configured otherwise.
    assert SleepPolicy.resolve(SleepConfig(enabled=True)).idle_seconds == 600


def test_existing_config_has_sleep_disabled() -> None:
    from shardon_core.config.loader import load_repository_config
    from sleep_fixtures import source_repo_root

    config = load_repository_config(source_repo_root() / "config")
    assert all(not config.sleep_policy_for(item).enabled for item in config.deployments)


def test_idle_sleep_only_after_idle_seconds(tmp_path, runtimes) -> None:
    runtime = _single(tmp_path, runtimes)
    assert asyncio.run(runtime.sleep_idle_deployments()) == []
    assert runtime.snapshot().deployments["dep-a"].state == "ready"

    _age_activity(runtime, "dep-a", 601)
    assert asyncio.run(runtime.sleep_idle_deployments()) == ["dep-a"]
    state = runtime.snapshot().deployments["dep-a"]
    assert state.state == "sleeping"
    assert state.loaded is True  # same process and port
    assert state.sleep_level == 1
    assert state.awake_memory_fraction == pytest.approx(0.45)
    assert state.resident_memory_fraction == pytest.approx(0.05)
    assert mock_stats(runtime, "dep-a")["sleeping"] is True
    assert events(runtime, "deployment.sleep")[-1]["data"]["source"] == "idle"


def test_health_probes_and_model_listing_never_wake_or_count_as_activity(
    tmp_path, runtimes
) -> None:
    runtime = _single(tmp_path, runtimes)
    _age_activity(runtime, "dep-a", 601)
    for _ in range(3):
        asyncio.run(runtime.refresh_backend_health())
        runtime.list_api_models()
    # Probes did not reset the idle clock.
    assert asyncio.run(runtime.sleep_idle_deployments()) == ["dep-a"]

    for _ in range(3):
        snapshot = asyncio.run(runtime.refresh_backend_health())
        runtime.list_api_models()
    assert snapshot.backend_health["dep-a"]["status"] == "sleeping"
    assert snapshot.deployments["dep-a"].state == "sleeping"
    stats = mock_stats(runtime, "dep-a")
    assert stats["wake_calls"] == 0
    assert stats["sleeping"] is True
    assert stats["health_calls"] >= 6


def test_request_wakes_sleeping_deployment_before_forwarding(tmp_path, runtimes) -> None:
    runtime = _single(tmp_path, runtimes)
    asyncio.run(runtime.sleep_deployment("dep-a", actor="test"))
    pid = runtime.snapshot().deployments["dep-a"].process_id

    response = asyncio.run(runtime.route_chat(_chat(), AUTH))

    assert response["object"] == "chat.completion"
    state = runtime.snapshot().deployments["dep-a"]
    assert state.state == "ready"
    assert state.process_id == pid  # woken, not restarted
    assert state.resident_memory_fraction == pytest.approx(0.45)
    stats = mock_stats(runtime, "dep-a")
    assert stats["wake_calls"] == 1
    assert stats["inference_while_sleeping"] == 0
    wake = events(runtime, "deployment.wake")[-1]["data"]
    assert wake["duration_seconds"] >= 0
    assert wake["trigger"].startswith("req_")


def test_concurrent_requests_share_a_single_wake(tmp_path, runtimes) -> None:
    runtime = _single(tmp_path, runtimes, environment={"MOCK_WAKE_DELAY_SECONDS": "0.5"})
    asyncio.run(runtime.sleep_deployment("dep-a", actor="test"))

    async def burst():
        return await asyncio.gather(*(runtime.route_chat(_chat(), AUTH) for _ in range(5)))

    responses = asyncio.run(burst())

    assert len(responses) == 5
    stats = mock_stats(runtime, "dep-a")
    assert stats["wake_calls"] == 1
    assert stats["inference_while_sleeping"] == 0
    assert len(events(runtime, "deployment.wake")) == 1


def test_level_two_wake_reloads_weights(tmp_path, runtimes) -> None:
    runtime = _single(tmp_path, runtimes, sleep={"level": 2})
    asyncio.run(runtime.sleep_deployment("dep-a", actor="test"))
    assert runtime.snapshot().deployments["dep-a"].sleep_level == 2

    asyncio.run(runtime.route_chat(_chat(), AUTH))

    stats = mock_stats(runtime, "dep-a")
    assert stats["wake_calls"] == 2  # tags=weights, then tags=kv_cache
    assert stats["reload_calls"] == 1
    assert stats["weights_loaded"] is True


def test_wake_failure_falls_back_to_cold_restart(tmp_path, runtimes) -> None:
    runtime = _single(tmp_path, runtimes, environment={"MOCK_WAKE_FAIL": "1"})
    asyncio.run(runtime.sleep_deployment("dep-a", actor="test"))
    old_pid = runtime.snapshot().deployments["dep-a"].process_id

    response = asyncio.run(runtime.route_chat(_chat(), AUTH))

    assert response["object"] == "chat.completion"
    state = runtime.snapshot().deployments["dep-a"]
    assert state.state == "ready"
    assert state.process_id != old_pid
    assert events(runtime, "deployment.wake_failed")
    assert events(runtime, "deployment.wake_fallback_restart")


def test_wake_time_counts_against_interactive_timeout(tmp_path, runtimes) -> None:
    runtime = _single(
        tmp_path,
        runtimes,
        environment={"MOCK_WAKE_DELAY_SECONDS": "2"},
        global_overrides={"interactive_request_timeout_seconds": 1},
    )
    asyncio.run(runtime.sleep_deployment("dep-a", actor="test"))

    async def scenario():
        with pytest.raises(RuntimeOperationError) as exc:
            await runtime.route_chat(_chat(), AUTH)
        # The shared wake keeps going for the next request.
        await runtime._wake_tasks["dep-a"]
        return exc.value

    error = asyncio.run(scenario())
    assert "timed out waiting for the deployment to wake" in str(error)
    assert runtime.snapshot().deployments["dep-a"].state == "ready"
    assert not runtime.snapshot().queued_requests


def test_batch_work_blocks_sleep_and_wakes_sleeping_deployment(tmp_path, runtimes) -> None:
    runtime = _single(tmp_path, runtimes)
    _age_activity(runtime, "dep-a", 601)
    job = BatchJobState(
        id="batch_1",
        api_key_id="k1",
        user_name="alice",
        model_name="model-dep-a",
        status="queued",
        created_at=utc_now_iso(),
        updated_at=utc_now_iso(),
        total_items=2,
        metadata={
            "requests": [
                {"model": "model-dep-a", "messages": [{"role": "user", "content": "1"}]},
                {"model": "model-dep-a", "messages": [{"role": "user", "content": "2"}]},
            ]
        },
    )
    runtime.state_store.mutate(lambda state: runtime._add_batch(state, job))
    # Queued batch work targeting the model keeps it awake.
    assert asyncio.run(runtime.sleep_idle_deployments()) == []
    with pytest.raises(RuntimeOperationError):
        asyncio.run(runtime.sleep_deployment("dep-a", actor="test"))

    # Force it asleep anyway (eviction path skips idle checks), then run batch.
    asyncio.run(runtime.sleep_deployment("dep-a", actor="test", require_idle=False))
    assert runtime.snapshot().deployments["dep-a"].state == "sleeping"
    asyncio.run(runtime.process_one_batch())

    assert runtime.snapshot().batch_jobs["batch_1"].status == "completed"
    stats = mock_stats(runtime, "dep-a")
    assert stats["wake_calls"] == 1
    assert stats["inference_while_sleeping"] == 0


def test_manual_sleep_refused_while_requests_are_active_or_queued(tmp_path, runtimes) -> None:
    runtime = _single(tmp_path, runtimes)
    queued = ActiveRequest(
        id="req-q",
        user_name="alice",
        api_key_id="k1",
        deployment_id="dep-a",
        backend_runtime_id="vllm-a",
        gpu_group_id="rtx",
        request_class="interactive",
        model_name="model-dep-a",
        status="queued",
        priority=100,
        created_at=utc_now_iso(),
    )
    runtime.state_store.mutate(lambda state: runtime._enqueue_request(state, queued))
    with pytest.raises(RuntimeOperationError) as exc:
        asyncio.run(runtime.sleep_deployment("dep-a", actor="test", source="public"))
    assert exc.value.detail["reason"] == "requests are queued for this deployment"

    def activate(state):
        state.queued_requests = []
        state.deployments["dep-a"].active_request_ids = ["req-live"]
        return state

    runtime.state_store.mutate(activate)
    with pytest.raises(RuntimeOperationError) as exc:
        asyncio.run(runtime.sleep_deployment("dep-a", actor="test"))
    assert exc.value.detail["reason"] == "deployment is serving requests"
    assert mock_stats(runtime, "dep-a")["sleep_calls"] == 0


def test_sleep_requires_opt_in(tmp_path, runtimes) -> None:
    repo = build_repo(
        tmp_path,
        backends=[mock_backend("vllm-a")],
        deployments=[deployment("dep-a", "vllm-a")],
    )
    runtime = make_runtime(repo)
    runtimes.append(runtime)
    asyncio.run(runtime.load_deployment(deployment_id="dep-a", actor="test"))
    with pytest.raises(RuntimeOperationError) as exc:
        asyncio.run(runtime.sleep_deployment("dep-a", actor="test"))
    assert exc.value.status_code == 409
    _age_activity(runtime, "dep-a", 10_000)
    assert asyncio.run(runtime.sleep_idle_deployments()) == []


def test_manual_wake_and_load_of_sleeping_deployment(tmp_path, runtimes) -> None:
    runtime = _single(tmp_path, runtimes)
    asyncio.run(runtime.sleep_deployment("dep-a", actor="test"))
    result = asyncio.run(runtime.load_deployment(deployment_id="dep-a", actor="test"))
    assert result["status"] == "ok"
    assert runtime.snapshot().deployments["dep-a"].state == "ready"
    again = asyncio.run(runtime.wake_deployment("dep-a", actor="test"))
    assert again["detail"] == "deployment already awake"


def _two_static(tmp_path, runtimes, *, shared_port: bool = False, group=None, sleep=None):
    sleep = {**SLEEP, "sleep_on_evict": True, **(sleep or {})}
    backend_a = mock_backend("vllm-a", sleep=sleep)
    backend_b = (
        {**backend_a, "id": "vllm-b", "display_name": "vllm-b"}
        if shared_port
        else mock_backend("vllm-b", sleep=sleep)
    )
    repo = build_repo(
        tmp_path,
        backends=[backend_a, backend_b],
        deployments=[deployment("dep-a", "vllm-a"), deployment("dep-b", "vllm-b")],
        group=group,
    )
    runtime = make_runtime(repo)
    runtimes.append(runtime)
    return runtime


def test_sleep_on_evict_keeps_previous_deployment_asleep(tmp_path, runtimes) -> None:
    runtime = _two_static(tmp_path, runtimes)
    asyncio.run(runtime.route_chat(_chat("model-dep-a"), AUTH))
    pid_a = runtime.snapshot().deployments["dep-a"].process_id

    asyncio.run(runtime.route_chat(_chat("model-dep-b"), AUTH))
    state = runtime.snapshot().deployments
    assert state["dep-a"].state == "sleeping"
    assert state["dep-a"].process_id == pid_a
    assert state["dep-b"].state == "ready"

    # Back to A: wake it and put B to sleep (static: one awake at a time).
    asyncio.run(runtime.route_chat(_chat("model-dep-a"), AUTH))
    state = runtime.snapshot().deployments
    assert state["dep-a"].state == "ready"
    assert state["dep-a"].process_id == pid_a
    assert state["dep-b"].state == "sleeping"
    assert mock_stats(runtime, "dep-a")["inference_while_sleeping"] == 0


def test_sleep_on_evict_stops_when_new_deployment_needs_the_same_port(tmp_path, runtimes) -> None:
    runtime = _two_static(tmp_path, runtimes, shared_port=True)
    asyncio.run(runtime.route_chat(_chat("model-dep-a"), AUTH))
    asyncio.run(runtime.route_chat(_chat("model-dep-b"), AUTH))
    state = runtime.snapshot().deployments
    assert state["dep-a"].loaded is False
    assert state["dep-a"].state == "unloaded"
    assert state["dep-b"].state == "ready"
    assert not events(runtime, "deployment.sleep")


def test_max_sleeping_per_group_stops_least_recently_used(tmp_path, runtimes) -> None:
    runtime = _two_static(tmp_path, runtimes, group={"max_sleeping": 1})
    asyncio.run(runtime.load_deployment(deployment_id="dep-a", actor="test"))
    asyncio.run(runtime.sleep_deployment("dep-a", actor="test"))
    # Loading B also wakes nothing; then B goes to sleep -> two sleepers, cap 1.
    asyncio.run(runtime.load_deployment(deployment_id="dep-b", actor="test"))
    asyncio.run(runtime.sleep_deployment("dep-b", actor="test"))
    state = runtime.snapshot().deployments
    assert state["dep-a"].loaded is False  # least recently used sleeper stopped
    assert state["dep-b"].state == "sleeping"
    stop = events(runtime, "deployment.sleep_stop")[-1]["data"]
    assert stop["deployment_id"] == "dep-a"
    assert "max_sleeping" in stop["reason"]


def test_global_max_sleeping_zero_means_sleep_then_stop(tmp_path, runtimes) -> None:
    runtime = _single(tmp_path, runtimes, global_overrides={"max_sleeping_deployments": 0})
    asyncio.run(runtime.sleep_deployment("dep-a", actor="test"))
    assert runtime.snapshot().deployments["dep-a"].loaded is False


def test_max_sleep_seconds_stops_long_sleepers(tmp_path, runtimes) -> None:
    runtime = _single(tmp_path, runtimes, sleep={"max_sleep_seconds": 3600})
    asyncio.run(runtime.sleep_deployment("dep-a", actor="test"))
    assert asyncio.run(runtime.enforce_sleep_limits()) == []

    def age(state):
        state.deployments["dep-a"].slept_at = (utc_now() - timedelta(hours=2)).isoformat()
        return state

    runtime.state_store.mutate(age)
    assert asyncio.run(runtime.enforce_sleep_limits()) == ["dep-a"]
    assert runtime.snapshot().deployments["dep-a"].state == "unloaded"


def test_failed_sleep_reverts_to_ready(tmp_path, runtimes) -> None:
    repo = build_repo(
        tmp_path,
        backends=[mock_backend("vllm-a", sleep=SLEEP, enable_sleep_mode=False)],
        deployments=[deployment("dep-a", "vllm-a")],
    )
    runtime = make_runtime(repo)
    runtimes.append(runtime)
    asyncio.run(runtime.load_deployment(deployment_id="dep-a", actor="test"))
    with pytest.raises(RuntimeOperationError):
        asyncio.run(runtime.sleep_deployment("dep-a", actor="test"))
    assert runtime.snapshot().deployments["dep-a"].state == "ready"
    assert events(runtime, "deployment.sleep_failed")
    asyncio.run(runtime.route_chat(_chat(), AUTH))
