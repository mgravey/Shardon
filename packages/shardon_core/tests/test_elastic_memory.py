from __future__ import annotations

import asyncio
import json
import socket
import sys
from datetime import timedelta

import pytest
from shardon_core.api.schemas import ChatCompletionRequest
from shardon_core.config.schemas import KvcachedConfig, parse_size_bytes
from shardon_core.scheduler.engine import SchedulingRequest
from shardon_core.services.observability import activity_snapshot
from shardon_core.services.runtime import RuntimeOperationError
from shardon_core.state.models import GPUProcessInfo
from shardon_core.utils.time import utc_now
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

ELASTIC = {"memory_model": "elastic", "usable_memory_fraction": 0.95}


@pytest.fixture
def runtimes():
    created = []
    yield created
    for runtime in created:
        shutdown(runtime)


def _free_port_range(size: int) -> tuple[int, int]:
    """Find ``size`` consecutive free ports."""
    for _ in range(50):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            first = probe.getsockname()[1]
        if first + size > 65535:
            continue
        ok = True
        for port in range(first, first + size):
            with socket.socket() as sock:
                try:
                    sock.bind(("127.0.0.1", port))
                except OSError:
                    ok = False
                    break
        if ok:
            return first, first + size - 1
    raise RuntimeError("no free port range")


def _ranged_backend(backend_id: str = "vllm-pool", *, size: int = 4, **kwargs):
    first, last = _free_port_range(size)
    return mock_backend(backend_id, port=first, port_range=[first, last], **kwargs)


def _runtime(tmp_path, runtimes, *, backends, deployments, group=ELASTIC, **kwargs):
    repo = build_repo(tmp_path, backends=backends, deployments=deployments, group=group)
    runtime = make_runtime(repo, **kwargs)
    runtimes.append(runtime)
    return runtime


def _chat(model: str) -> ChatCompletionRequest:
    return ChatCompletionRequest(model=model, messages=[{"role": "user", "content": "hi"}])


def _three(tmp_path, runtimes, *, sleep=None, group=ELASTIC, **provider_kwargs):
    backend = _ranged_backend(sleep=sleep)
    return _runtime(
        tmp_path,
        runtimes,
        backends=[backend],
        deployments=[
            deployment(name, backend["id"], resident_memory_fraction=0.4)
            for name in ("dep-a", "dep-b", "dep-c")
        ],
        group=group,
        **provider_kwargs,
    )


def test_size_parsing_and_limit_fraction() -> None:
    assert parse_size_bytes("512M") == 512 * 1024**2
    assert parse_size_bytes("24G") == 24 * 1024**3
    assert parse_size_bytes("100_000") == 100_000
    with pytest.raises(ValueError):
        KvcachedConfig(limit="lots")
    assert KvcachedConfig(limit="24G").limit_fraction(96 * 1024**3) == pytest.approx(0.25)
    assert KvcachedConfig(limit_percent=30).limit_fraction(1) == pytest.approx(0.3)
    assert KvcachedConfig().limit_fraction(1) == 0.0


def test_two_deployments_co_resident_in_elastic_group(tmp_path, runtimes) -> None:
    runtime = _three(tmp_path, runtimes)
    first, _ = runtime.config.backends["vllm-pool"].port_range

    asyncio.run(runtime.route_chat(_chat("model-dep-a"), AUTH))
    asyncio.run(runtime.route_chat(_chat("model-dep-b"), AUTH))

    state = runtime.snapshot().deployments
    assert state["dep-a"].state == "ready" and state["dep-b"].state == "ready"
    assert {state["dep-a"].port, state["dep-b"].port} == {first, first + 1}
    assert not events(runtime, "deployment.evict")
    a, b = mock_stats(runtime, "dep-a"), mock_stats(runtime, "dep-b")
    assert a["pid"] != b["pid"]
    assert a["runtime_port"] == str(state["dep-a"].port)
    assert b["runtime_port"] == str(state["dep-b"].port)
    assert a["deployment_id"] == "dep-a" and b["deployment_id"] == "dep-b"
    assert a["inference_calls"] == 1 and b["inference_calls"] == 1
    # A request routes to its own process, not the shared base_url port.
    asyncio.run(runtime.route_chat(_chat("model-dep-a"), AUTH))
    assert mock_stats(runtime, "dep-a")["inference_calls"] == 2
    assert mock_stats(runtime, "dep-b")["inference_calls"] == 1


def test_elastic_evicts_least_recently_used_when_footprints_exceed_limit(
    tmp_path, runtimes
) -> None:
    runtime = _three(tmp_path, runtimes)
    asyncio.run(runtime.route_chat(_chat("model-dep-a"), AUTH))
    asyncio.run(runtime.route_chat(_chat("model-dep-b"), AUTH))
    asyncio.run(runtime.route_chat(_chat("model-dep-a"), AUTH))  # B is now LRU

    asyncio.run(runtime.route_chat(_chat("model-dep-c"), AUTH))

    state = runtime.snapshot().deployments
    assert state["dep-a"].state == "ready"
    assert state["dep-b"].loaded is False
    assert state["dep-c"].state == "ready"
    assert [item["data"]["deployment_id"] for item in events(runtime, "deployment.evict")] == [
        "dep-b"
    ]


def test_elastic_refuses_admission_when_busy_deployments_hold_the_memory(
    tmp_path, runtimes
) -> None:
    runtime = _three(tmp_path, runtimes)
    asyncio.run(runtime.load_deployment(deployment_id="dep-a", actor="test"))
    asyncio.run(runtime.load_deployment(deployment_id="dep-b", actor="test"))

    def busy(state):
        state.deployments["dep-a"].active_request_ids = ["r1"]
        state.deployments["dep-b"].active_request_ids = ["r2"]
        return state

    runtime.state_store.mutate(busy)
    snapshot = runtime.refresh_gpu_observations()
    decision = runtime.scheduler.schedule(
        SchedulingRequest(
            model_name="model-dep-c",
            task="chat",
            priority=100,
            request_class="interactive",
            request_id="r3",
        ),
        snapshot,
        utc_now(),
    )
    assert decision.accepted is False
    assert decision.status_code == 409
    with pytest.raises(RuntimeOperationError):
        asyncio.run(runtime.load_deployment(deployment_id="dep-c", actor="test"))


def test_elastic_sleeps_idle_deployments_before_stopping(tmp_path, runtimes) -> None:
    runtime = _three(
        tmp_path, runtimes, sleep={"enabled": True, "sleep_on_evict": True}
    )
    asyncio.run(runtime.route_chat(_chat("model-dep-a"), AUTH))
    asyncio.run(runtime.route_chat(_chat("model-dep-b"), AUTH))

    asyncio.run(runtime.route_chat(_chat("model-dep-c"), AUTH))

    state = runtime.snapshot().deployments
    assert state["dep-a"].state == "sleeping"  # LRU, slept rather than stopped
    assert state["dep-a"].port is not None  # a sleeper keeps its port
    assert state["dep-b"].state == "ready"
    assert state["dep-c"].state == "ready"
    assert len({state[name].port for name in ("dep-a", "dep-b", "dep-c")}) == 3
    assert not events(runtime, "deployment.evict")


def test_elastic_fits_external_gpu_usage_via_observed_free_memory(tmp_path, runtimes) -> None:
    # Only ~30% of the card is physically free (e.g. WhisperX/TTS outside Shardon).
    runtime = _three(tmp_path, runtimes, free_memory_mb=15_000, total_memory_mb=49_152)
    with pytest.raises(RuntimeOperationError):
        asyncio.run(runtime.load_deployment(deployment_id="dep-a", actor="test"))
    assert runtime.snapshot().deployments["dep-a"].loaded is False


def test_port_allocation_lifecycle(tmp_path, runtimes) -> None:
    runtime = _three(tmp_path, runtimes)
    first, _ = runtime.config.backends["vllm-pool"].port_range
    asyncio.run(runtime.load_deployment(deployment_id="dep-a", actor="test"))
    asyncio.run(runtime.load_deployment(deployment_id="dep-b", actor="test"))
    assert runtime.snapshot().deployments["dep-a"].port == first
    assert runtime.snapshot().deployments["dep-b"].port == first + 1

    asyncio.run(runtime.unload_deployment("dep-a", actor="test"))
    assert runtime.snapshot().deployments["dep-a"].port is None  # released

    asyncio.run(runtime.load_deployment(deployment_id="dep-c", actor="test"))
    assert runtime.snapshot().deployments["dep-c"].port == first  # reused
    assert mock_stats(runtime, "dep-c")["runtime_port"] == str(first)


def test_port_range_exhaustion_fails_the_start_cleanly(tmp_path, runtimes) -> None:
    backend = _ranged_backend(size=1)
    runtime = _runtime(
        tmp_path,
        runtimes,
        backends=[backend],
        deployments=[
            deployment(name, backend["id"], resident_memory_fraction=0.3)
            for name in ("dep-a", "dep-b")
        ],
    )
    asyncio.run(runtime.load_deployment(deployment_id="dep-a", actor="test"))
    with pytest.raises(RuntimeOperationError) as exc:
        asyncio.run(runtime.load_deployment(deployment_id="dep-b", actor="test"))
    assert "no free port" in json.dumps(exc.value.detail)
    state = runtime.snapshot().deployments
    assert state["dep-a"].state == "ready"
    assert state["dep-b"].state == "failed" and state["dep-b"].port is None


def test_fixed_deployment_port_conflict_forces_stop_even_in_elastic_group(
    tmp_path, runtimes
) -> None:
    first, _ = _free_port_range(2)
    backend = mock_backend("vllm-pool", port=first, port_range=[first, first + 1])
    runtime = _runtime(
        tmp_path,
        runtimes,
        backends=[backend],
        deployments=[
            deployment("dep-a", "vllm-pool", resident_memory_fraction=0.2, port=first),
            deployment("dep-b", "vllm-pool", resident_memory_fraction=0.2, port=first),
        ],
    )
    asyncio.run(runtime.route_chat(_chat("model-dep-a"), AUTH))
    asyncio.run(runtime.route_chat(_chat("model-dep-b"), AUTH))
    state = runtime.snapshot().deployments
    assert state["dep-a"].loaded is False
    assert state["dep-b"].port == first


def test_static_group_behaviour_is_unchanged(tmp_path, runtimes) -> None:
    runtime = _three(tmp_path, runtimes, group={})
    assert runtime.config.gpu_groups["rtx"].memory_model == "static"
    asyncio.run(runtime.route_chat(_chat("model-dep-a"), AUTH))
    asyncio.run(runtime.route_chat(_chat("model-dep-b"), AUTH))
    state = runtime.snapshot().deployments
    # One model at a time, even though the ports differ and memory would fit.
    assert state["dep-a"].loaded is False
    assert state["dep-b"].state == "ready"
    assert state["dep-b"].resident_memory_fraction == pytest.approx(0.45)
    assert state["dep-b"].footprint_source == "config"


def test_footprint_learned_from_per_process_gpu_memory(tmp_path, runtimes) -> None:
    backend = _ranged_backend(kvcached={"enabled": True, "kvctl_path": "/nonexistent/kvctl"})
    runtime = _runtime(
        tmp_path,
        runtimes,
        backends=[backend],
        deployments=[
            deployment(
                "dep-a",
                backend["id"],
                resident_memory_fraction=0.6,
                kvcached={"limit": "12G"},
            )
        ],
        total_memory_mb=49_152,
    )
    asyncio.run(runtime.load_deployment(deployment_id="dep-a", actor="test"))
    state = runtime.snapshot().deployments["dep-a"]
    assert state.footprint_source == "estimate"
    assert state.resident_memory_fraction == pytest.approx(0.6)

    runtime.gpu_provider.set_processes(
        [GPUProcessInfo(pid=state.process_id, user_name="svc", gpu_id="gpu0", command="vllm",
                        memory_mb=12_288)]
    )
    asyncio.run(runtime.refresh_elastic_footprints())
    state = runtime.snapshot().deployments["dep-a"]
    assert state.footprint_source == "observed"
    assert state.observed_memory_mb == 12_288
    # 12 GiB weights/runtime + 12 GiB kvcached limit on a 48 GiB card.
    assert state.resident_memory_fraction == pytest.approx(0.5)

    # The learned footprint is the next estimate after a stop.
    asyncio.run(runtime.unload_deployment("dep-a", actor="test"))
    runtime.gpu_provider.set_processes([])
    asyncio.run(runtime.load_deployment(deployment_id="dep-a", actor="test"))
    assert runtime.snapshot().deployments["dep-a"].resident_memory_fraction == pytest.approx(0.5)


def _fake_kvctl(tmp_path) -> tuple[str, object]:
    log = tmp_path / "kvctl.log"
    script = tmp_path / "kvctl"
    script.write_text(
        f"""#!{sys.executable}
import json, sys
with open({str(log)!r}, "a") as handle:
    handle.write(json.dumps(sys.argv[1:]) + "\\n")
if sys.argv[1] == "list":
    names = [arg for arg in sys.argv[2:] if not arg.startswith("--")]
    print(json.dumps([
        {{"ipc": name, "limit_bytes": 4 * 1024**3, "used_bytes": 123456, "gpus": []}}
        for name in names
    ]))
""",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return str(script), log


def test_kvcached_controls_ipc_name_and_live_usage(tmp_path, runtimes) -> None:
    kvctl, log = _fake_kvctl(tmp_path)
    backend = _ranged_backend(
        environment={"ENABLE_KVCACHED": "true", "KVCACHED_AUTOPATCH": "1"},
        kvcached={"kvctl_path": kvctl, "limit": "4G", "min": "1G", "priority": 5,
                  "headroom": "512M"},
    )
    runtime = _runtime(
        tmp_path,
        runtimes,
        backends=[backend],
        deployments=[
            deployment("dep-a", backend["id"], resident_memory_fraction=0.3,
                       kvcached={"priority": 10}),
        ],
    )
    asyncio.run(runtime.load_deployment(deployment_id="dep-a", actor="test"))

    assert runtime.snapshot().deployments["dep-a"].kvcached_ipc_name == "igf-shardon-dep-a"
    assert mock_stats(runtime, "dep-a")["kvcached_ipc_name"] == "igf-shardon-dep-a"
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert calls == [
        ["limit", "igf-shardon-dep-a", "4G"],
        ["min", "igf-shardon-dep-a", "1G"],
        ["priority", "igf-shardon-dep-a", "10"],
        ["headroom", "igf-shardon-dep-a", "512M"],
    ]

    usage = asyncio.run(runtime.refresh_kvcached_usage())
    assert usage["dep-a"]["used_bytes"] == 123456
    assert runtime.status()["kvcached_usage"]["dep-a"]["limit_bytes"] == 4 * 1024**3
    activity = activity_snapshot(runtime.snapshot(), runtime.config)
    row = activity.models[0].deployments[0]
    assert row.kv_used_bytes == 123456
    assert row.port == runtime.snapshot().deployments["dep-a"].port


def test_configured_kvcached_ipc_name_is_kept(tmp_path, runtimes) -> None:
    backend = _ranged_backend(
        environment={"ENABLE_KVCACHED": "true", "KVCACHED_IPC_NAME": "operator-chosen"},
        kvcached={"kvctl_path": "/nonexistent/kvctl"},
    )
    runtime = _runtime(
        tmp_path,
        runtimes,
        backends=[backend],
        deployments=[deployment("dep-a", backend["id"], resident_memory_fraction=0.3)],
    )
    asyncio.run(runtime.load_deployment(deployment_id="dep-a", actor="test"))
    assert runtime.snapshot().deployments["dep-a"].kvcached_ipc_name == "operator-chosen"
    assert mock_stats(runtime, "dep-a")["kvcached_ipc_name"] == "operator-chosen"


def test_missing_kvctl_never_fails_the_deployment(tmp_path, runtimes) -> None:
    backend = _ranged_backend(kvcached={"kvctl_path": "/nonexistent/kvctl", "limit": "2G"})
    runtime = _runtime(
        tmp_path,
        runtimes,
        backends=[backend],
        deployments=[deployment("dep-a", backend["id"], resident_memory_fraction=0.3)],
    )
    asyncio.run(runtime.load_deployment(deployment_id="dep-a", actor="test"))
    assert runtime.snapshot().deployments["dep-a"].state == "ready"
    assert events(runtime, "kvcached.apply_failed")
    usage = asyncio.run(runtime.refresh_kvcached_usage())
    assert "cannot run" in usage["dep-a"]["error"]


def test_no_kvcached_env_without_opt_in(tmp_path, runtimes) -> None:
    backend = _ranged_backend()
    runtime = _runtime(
        tmp_path,
        runtimes,
        backends=[backend],
        deployments=[deployment("dep-a", backend["id"], resident_memory_fraction=0.3)],
    )
    asyncio.run(runtime.load_deployment(deployment_id="dep-a", actor="test"))
    assert runtime.snapshot().deployments["dep-a"].kvcached_ipc_name is None
    assert mock_stats(runtime, "dep-a")["kvcached_ipc_name"] is None


def test_wake_in_elastic_group_reclaims_memory_from_idle_peers(tmp_path, runtimes) -> None:
    runtime = _three(
        tmp_path, runtimes, sleep={"enabled": True, "sleep_on_evict": True}
    )
    for name in ("a", "b", "c"):  # C's arrival puts A (LRU) to sleep
        asyncio.run(runtime.route_chat(_chat(f"model-dep-{name}"), AUTH))
    assert runtime.snapshot().deployments["dep-a"].state == "sleeping"

    def age(state):
        state.deployments["dep-b"].last_used_at = (utc_now() - timedelta(hours=1)).isoformat()
        return state

    runtime.state_store.mutate(age)
    asyncio.run(runtime.route_chat(_chat("model-dep-a"), AUTH))

    state = runtime.snapshot().deployments
    assert state["dep-a"].state == "ready"
    assert state["dep-b"].state == "sleeping"  # the LRU idle peer made room
    assert state["dep-c"].state == "ready"
    assert mock_stats(runtime, "dep-a")["inference_while_sleeping"] == 0


def test_per_deployment_ports_start_at_base_url_port_and_increment(tmp_path, runtimes) -> None:
    first, _ = _free_port_range(3)
    backend = mock_backend("vllm-pool", port=first, per_deployment_ports=True)
    runtime = _runtime(
        tmp_path,
        runtimes,
        backends=[backend],
        deployments=[
            deployment(name, "vllm-pool", resident_memory_fraction=0.3)
            for name in ("dep-a", "dep-b", "dep-c")
        ],
    )
    assert runtime.config.backends["vllm-pool"].effective_port_range() == (first, first + 15)
    for name in ("dep-a", "dep-b", "dep-c"):
        asyncio.run(runtime.load_deployment(deployment_id=name, actor="test"))
    state = runtime.snapshot().deployments
    ports = {name: state[name].port for name in ("dep-a", "dep-b", "dep-c")}
    assert ports == {"dep-a": first, "dep-b": first + 1, "dep-c": first + 2}
    assert mock_stats(runtime, "dep-b")["runtime_port"] == str(first + 1)
