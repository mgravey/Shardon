from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class ActiveRequest(BaseModel):
    id: str
    router_instance_id: str | None = None
    interruption_reason: str | None = None
    user_name: str
    api_key_id: str
    deployment_id: str
    backend_runtime_id: str
    gpu_group_id: str
    request_class: Literal["interactive", "batch"]
    model_name: str
    status: Literal["queued", "running", "completed", "failed", "cancelled"]
    priority: int
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    detail: dict[str, Any] = Field(default_factory=dict)
    task: str | None = None
    requested_model_name: str | None = None
    phase: Literal["waiting", "loading", "in_flight"] = "waiting"
    wait_reason: str | None = None
    batch_id: str | None = None


class BatchJobState(BaseModel):
    id: str
    router_instance_id: str | None = None
    interruption_reason: str | None = None
    api_key_id: str
    user_name: str
    deployment_id: str | None = None
    model_name: str
    status: Literal["queued", "running", "completed", "failed", "cancelled"]
    created_at: str
    updated_at: str
    total_items: int
    gpu_group_id: str | None = None
    current_item: int | None = None
    cancelled_items: int = 0
    completed_items: int = 0
    failed_items: int = 0
    output_file_id: str | None = None
    error_file_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DrainState(BaseModel):
    gpu_group_id: str
    status: Literal["pending", "completed", "forced", "failed"]
    started_at: str
    timeout_seconds: int
    completed_at: str | None = None
    reason: str | None = None


DeploymentLifecycleState = Literal[
    "unloaded",
    "starting",
    "ready",
    "falling_asleep",
    "sleeping",
    "waking",
    "stopping",
    "failed",
]

# A process in one of these states is still loaded (same PID and port) but
# must be woken before any inference request or batch item is forwarded.
SLEEP_STATES: frozenset[str] = frozenset({"falling_asleep", "sleeping", "waking"})


class DeploymentRuntimeState(BaseModel):
    deployment_id: str
    gpu_group_id: str
    eligible_gpu_group_ids: list[str] = Field(default_factory=list)
    selected_gpu_group_id: str | None = None
    backend_runtime_id: str
    loaded: bool = False
    state: DeploymentLifecycleState = "unloaded"
    desired_state: Literal["loaded", "unloaded"] = "unloaded"
    loaded_at: str | None = None
    startup_started_at: str | None = None
    readiness_passed_at: str | None = None
    last_used_at: str | None = None
    resident_memory_fraction: float = 0.0
    active_request_ids: list[str] = Field(default_factory=list)
    keep_free_killed_at: str | None = None
    process_id: int | None = None
    # Router instance whose process launched process_id. Backends are children
    # of that router, so they stop with it; a dead backend whose owner has also
    # exited was stopped by a router restart, not by a crash.
    process_owner_instance_id: str | None = None
    current_model_name: str | None = None
    last_error: str | None = None
    last_transition_reason: str | None = None
    last_readiness_detail: dict[str, Any] = Field(default_factory=dict)
    # Sleep lifecycle (vLLM sleep mode). resident_memory_fraction reflects the
    # current footprint; awake_memory_fraction is restored on wake.
    sleep_level: int | None = None
    sleep_transition_started_at: str | None = None
    slept_at: str | None = None
    woke_at: str | None = None
    awake_memory_fraction: float | None = None
    last_sleep_detail: dict[str, Any] = Field(default_factory=dict)
    # Per-deployment port (backend port_range or deployment port); None means
    # the backend base_url port is used unchanged.
    port: int | None = None
    kvcached_ipc_name: str | None = None
    # Elastic memory: where resident_memory_fraction came from, and the
    # per-process GPU memory observed for this deployment's process tree.
    footprint_source: Literal["config", "estimate", "observed"] = "config"
    observed_memory_mb: int | None = None
    # Last observed awake footprint; survives unloads so the next admission
    # in the same group starts from a measured value instead of an estimate.
    learned_memory_fraction: float | None = None
    learned_gpu_group_id: str | None = None

    @property
    def is_asleep(self) -> bool:
        return self.loaded and self.state in SLEEP_STATES

    def last_activity_at(self) -> datetime | None:
        """Latest real activity. Health probes never update these fields."""
        stamps: list[datetime] = []
        for raw in (self.last_used_at, self.woke_at, self.readiness_passed_at, self.loaded_at):
            if not raw:
                continue
            try:
                stamp = datetime.fromisoformat(raw)
            except ValueError:
                continue
            stamps.append(stamp if stamp.tzinfo else stamp.replace(tzinfo=UTC))
        return max(stamps, default=None)


class GPUProcessInfo(BaseModel):
    pid: int
    user_name: str
    gpu_id: str
    command: str
    memory_mb: int = 0
    managed_by_shardon: bool = False


class GPUObservation(BaseModel):
    gpu_id: str
    free_memory_mb: int
    total_memory_mb: int
    observed_processes: list[GPUProcessInfo] = Field(default_factory=list)


class RuntimeStateSnapshot(BaseModel):
    deployments: dict[str, DeploymentRuntimeState] = Field(default_factory=dict)
    active_requests: dict[str, ActiveRequest] = Field(default_factory=dict)
    queued_requests: list[ActiveRequest] = Field(default_factory=list)
    batch_jobs: dict[str, BatchJobState] = Field(default_factory=dict)
    drains: dict[str, DrainState] = Field(default_factory=dict)
    gpu_observations: dict[str, GPUObservation] = Field(default_factory=dict)
    backend_health: dict[str, dict[str, Any]] = Field(default_factory=dict)
    # Live kvcached usage per deployment (from `kvctl list --json`).
    kvcached_usage: dict[str, dict[str, Any]] = Field(default_factory=dict)
