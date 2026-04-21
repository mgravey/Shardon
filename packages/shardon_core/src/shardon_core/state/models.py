from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ActiveRequest(BaseModel):
    id: str
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


class BatchJobState(BaseModel):
    id: str
    api_key_id: str
    user_name: str
    deployment_id: str | None = None
    model_name: str
    status: Literal["queued", "running", "completed", "failed", "cancelled"]
    created_at: str
    updated_at: str
    total_items: int
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


class DeploymentRuntimeState(BaseModel):
    deployment_id: str
    gpu_group_id: str
    backend_runtime_id: str
    loaded: bool = False
    state: Literal["unloaded", "starting", "ready", "stopping", "failed"] = "unloaded"
    desired_state: Literal["loaded", "unloaded"] = "unloaded"
    loaded_at: str | None = None
    startup_started_at: str | None = None
    readiness_passed_at: str | None = None
    last_used_at: str | None = None
    resident_memory_fraction: float = 0.0
    active_request_ids: list[str] = Field(default_factory=list)
    keep_free_killed_at: str | None = None
    process_id: int | None = None
    current_model_name: str | None = None
    last_error: str | None = None
    last_transition_reason: str | None = None
    last_readiness_detail: dict[str, Any] = Field(default_factory=dict)


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
