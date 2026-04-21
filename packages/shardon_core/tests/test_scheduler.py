from datetime import UTC, datetime, timedelta

from shardon_core.config.schemas import (
    AdminUserRecord,
    BackendRuntimeConfig,
    DeploymentConfig,
    GPUDeviceConfig,
    GPUGroupConfig,
    GlobalConfig,
    ModelConfig,
    RepositoryConfig,
)
from shardon_core.scheduler.engine import SchedulerEngine, SchedulingRequest
from shardon_core.state.models import ActiveRequest, DeploymentRuntimeState, GPUObservation, RuntimeStateSnapshot


def _config() -> RepositoryConfig:
    return RepositoryConfig(
        global_config=GlobalConfig(switch_grace_window_seconds=300),
        backends={
            "backend-a": BackendRuntimeConfig(
                id="backend-a",
                backend_type="vllm",
                version="0.6",
                display_name="A",
                runtime_dir=".",
                base_url="http://127.0.0.1:1",
                launch_command=["python3", "-c", "print('a')"],
            ),
            "backend-b": BackendRuntimeConfig(
                id="backend-b",
                backend_type="vllm",
                version="0.7",
                display_name="B",
                runtime_dir=".",
                base_url="http://127.0.0.1:2",
                launch_command=["python3", "-c", "print('b')"],
            ),
        },
        models={
            "model-a": ModelConfig(
                id="model-a",
                source="/a",
                display_name="A",
                backend_compatibility=["vllm"],
                tasks=["chat"],
            ),
            "model-b": ModelConfig(
                id="model-b",
                source="/b",
                display_name="B",
                backend_compatibility=["vllm"],
                tasks=["chat"],
            ),
        },
        deployments={
            "dep-a": DeploymentConfig(
                id="dep-a",
                model_id="model-a",
                backend_runtime_id="backend-a",
                gpu_group_id="group-1",
                api_model_name="alpha",
                display_name="alpha",
                memory_fraction=0.45,
                tasks=["chat"],
            ),
            "dep-b": DeploymentConfig(
                id="dep-b",
                model_id="model-b",
                backend_runtime_id="backend-b",
                gpu_group_id="group-1",
                api_model_name="beta",
                display_name="beta",
                memory_fraction=0.45,
                tasks=["chat"],
            ),
        },
        gpu_devices={"gpu0": GPUDeviceConfig(id="gpu0")},
        gpu_groups={
            "group-1": GPUGroupConfig(
                id="group-1",
                display_name="Group 1",
                gpu_ids=["gpu0"],
                usable_memory_fraction=0.95,
            )
        },
        admin_users={
            "admin": AdminUserRecord(
                username="admin",
                password_hash="x",
                created_at="2026-04-21T00:00:00+00:00",
            )
        },
    )


def test_scheduler_prefers_loaded_compatible_deployment() -> None:
    scheduler = SchedulerEngine(_config())
    snapshot = RuntimeStateSnapshot(
        deployments={
            "dep-a": DeploymentRuntimeState(
                deployment_id="dep-a",
                gpu_group_id="group-1",
                backend_runtime_id="backend-a",
                loaded=True,
                last_used_at="2026-04-21T00:00:00+00:00",
            )
        }
    )
    decision = scheduler.schedule(
        SchedulingRequest("alpha", "chat", 100, "interactive", "req-1"),
        snapshot,
        datetime.now(tz=UTC),
    )
    assert decision.accepted is True
    assert decision.deployment_id == "dep-a"
    assert decision.should_load is False


def test_scheduler_holds_switch_during_grace_window() -> None:
    scheduler = SchedulerEngine(_config())
    created_at = (datetime.now(tz=UTC) - timedelta(seconds=30)).isoformat()
    snapshot = RuntimeStateSnapshot(
        deployments={
            "dep-a": DeploymentRuntimeState(
                deployment_id="dep-a",
                gpu_group_id="group-1",
                backend_runtime_id="backend-a",
                loaded=True,
                resident_memory_fraction=0.45,
            )
        },
        queued_requests=[
            ActiveRequest(
                id="queued-1",
                user_name="low",
                api_key_id="k1",
                deployment_id="dep-a",
                backend_runtime_id="backend-a",
                gpu_group_id="group-1",
                request_class="interactive",
                model_name="alpha",
                status="queued",
                priority=50,
                created_at=created_at,
            )
        ],
    )
    decision = scheduler.schedule(
        SchedulingRequest("beta", "chat", 200, "interactive", "req-2"),
        snapshot,
        datetime.now(tz=UTC),
    )
    assert decision.accepted is False
    assert decision.status_code == 409


def test_scheduler_switches_after_grace_window_expires() -> None:
    scheduler = SchedulerEngine(_config())
    created_at = (datetime.now(tz=UTC) - timedelta(seconds=360)).isoformat()
    snapshot = RuntimeStateSnapshot(
        deployments={
            "dep-a": DeploymentRuntimeState(
                deployment_id="dep-a",
                gpu_group_id="group-1",
                backend_runtime_id="backend-a",
                loaded=True,
                resident_memory_fraction=0.45,
            )
        },
        queued_requests=[
            ActiveRequest(
                id="queued-1",
                user_name="low",
                api_key_id="k1",
                deployment_id="dep-a",
                backend_runtime_id="backend-a",
                gpu_group_id="group-1",
                request_class="interactive",
                model_name="alpha",
                status="queued",
                priority=50,
                created_at=created_at,
            )
        ],
    )
    decision = scheduler.schedule(
        SchedulingRequest("beta", "chat", 200, "interactive", "req-2"),
        snapshot,
        datetime.now(tz=UTC),
    )
    assert decision.accepted is True
    assert decision.deployment_id == "dep-b"
    assert decision.should_load is True


def test_scheduler_allows_swap_when_eviction_reclaims_group_budget() -> None:
    config = _config()
    config.deployments["dep-a"].memory_fraction = 0.9
    config.deployments["dep-b"].memory_fraction = 0.9
    scheduler = SchedulerEngine(config)
    snapshot = RuntimeStateSnapshot(
        deployments={
            "dep-a": DeploymentRuntimeState(
                deployment_id="dep-a",
                gpu_group_id="group-1",
                backend_runtime_id="backend-a",
                loaded=True,
                resident_memory_fraction=0.9,
                last_used_at="2026-04-21T00:00:00+00:00",
            )
        }
    )
    decision = scheduler.schedule(
        SchedulingRequest("beta", "chat", 200, "interactive", "req-3"),
        snapshot,
        datetime.now(tz=UTC),
    )
    assert decision.accepted is True
    assert decision.deployment_id == "dep-b"
    assert decision.should_evict == ["dep-a"]


def test_scheduler_allows_swap_when_eviction_restores_free_memory_observation() -> None:
    config = _config()
    config.deployments["dep-a"].memory_fraction = 0.7
    config.deployments["dep-b"].memory_fraction = 0.7
    scheduler = SchedulerEngine(config)
    snapshot = RuntimeStateSnapshot(
        deployments={
            "dep-a": DeploymentRuntimeState(
                deployment_id="dep-a",
                gpu_group_id="group-1",
                backend_runtime_id="backend-a",
                loaded=True,
                resident_memory_fraction=0.7,
                last_used_at="2026-04-21T00:00:00+00:00",
            )
        },
        gpu_observations={
            "gpu0": GPUObservation(
                gpu_id="gpu0",
                free_memory_mb=100,
                total_memory_mb=1000,
            )
        },
    )
    decision = scheduler.schedule(
        SchedulingRequest("beta", "chat", 200, "interactive", "req-4"),
        snapshot,
        datetime.now(tz=UTC),
    )
    assert decision.accepted is True
    assert decision.deployment_id == "dep-b"
    assert decision.should_evict == ["dep-a"]
