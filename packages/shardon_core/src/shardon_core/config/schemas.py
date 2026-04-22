from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class GlobalConfig(BaseModel):
    instance_name: str = "shardon"
    admin_host: str = "127.0.0.1"
    admin_port: int = 8081
    router_host: str = "127.0.0.1"
    router_port: int = 8080
    state_root: str = "state"
    switch_grace_window_seconds: int = 300
    default_drain_timeout_seconds: int = 120
    scheduler_tick_seconds: int = 5
    queue_poll_interval_seconds: float = 0.25
    default_memory_fraction: float = 0.9
    interactive_request_timeout_seconds: int | None = None
    backend_startup_timeout_seconds: int | None = None
    backend_readiness_poll_interval_seconds: float = 1.0
    backend_stop_timeout_seconds: int = 30

    def effective_interactive_request_timeout_seconds(self) -> int:
        if self.interactive_request_timeout_seconds is not None:
            return self.interactive_request_timeout_seconds
        return 300 + self.switch_grace_window_seconds

    def effective_backend_startup_timeout_seconds(self) -> int:
        if self.backend_startup_timeout_seconds is not None:
            return self.backend_startup_timeout_seconds
        return 300 + self.switch_grace_window_seconds


class BackendCapabilities(BaseModel):
    chat: bool = True
    completions: bool = True
    embeddings: bool = False
    batch: bool = True
    openai_compatible: bool = True
    max_context_length: int | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class BackendRuntimeConfig(BaseModel):
    id: str
    backend_type: Literal["vllm", "sglang", "mock"]
    version: str
    display_name: str
    runtime_dir: str
    base_url: str
    launch_command: list[str]
    working_directory: str | None = None
    environment: dict[str, str] = Field(default_factory=dict)
    health_path: str = "/health"
    startup_timeout_seconds: int | None = None
    readiness_poll_interval_seconds: float | None = None
    stop_timeout_seconds: int | None = None
    gpu_group_overrides: dict[str, "BackendRuntimeOverride"] = Field(default_factory=dict)
    capabilities: BackendCapabilities = Field(default_factory=BackendCapabilities)

    def resolved_for_gpu_group(self, gpu_group_id: str | None) -> "BackendRuntimeConfig":
        if gpu_group_id is None:
            return self
        override = self.gpu_group_overrides.get(gpu_group_id)
        if override is None:
            return self
        environment = dict(self.environment)
        environment.update(override.environment)
        return self.model_copy(
            update={
                "base_url": override.base_url or self.base_url,
                "launch_command": override.launch_command or self.launch_command,
                "working_directory": (
                    override.working_directory
                    if override.working_directory is not None
                    else self.working_directory
                ),
                "health_path": override.health_path or self.health_path,
                "startup_timeout_seconds": (
                    override.startup_timeout_seconds
                    if override.startup_timeout_seconds is not None
                    else self.startup_timeout_seconds
                ),
                "readiness_poll_interval_seconds": (
                    override.readiness_poll_interval_seconds
                    if override.readiness_poll_interval_seconds is not None
                    else self.readiness_poll_interval_seconds
                ),
                "stop_timeout_seconds": (
                    override.stop_timeout_seconds
                    if override.stop_timeout_seconds is not None
                    else self.stop_timeout_seconds
                ),
                "environment": environment,
            }
        )


class BackendRuntimeOverride(BaseModel):
    base_url: str | None = None
    launch_command: list[str] | None = None
    working_directory: str | None = None
    environment: dict[str, str] = Field(default_factory=dict)
    health_path: str | None = None
    startup_timeout_seconds: int | None = None
    readiness_poll_interval_seconds: float | None = None
    stop_timeout_seconds: int | None = None


class ModelConfig(BaseModel):
    id: str
    source: str
    display_name: str
    backend_compatibility: list[str]
    tasks: list[Literal["chat", "completion", "embedding"]] = Field(default_factory=list)
    model_capabilities: list[Literal["text", "audio", "image", "video"]] = Field(
        default_factory=lambda: ["text"]
    )
    tokenizer: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def normalize_model_capabilities(self) -> "ModelConfig":
        deduped: list[Literal["text", "audio", "image", "video"]] = []
        for capability in self.model_capabilities:
            if capability not in deduped:
                deduped.append(capability)
        self.model_capabilities = deduped or ["text"]
        return self


class DeploymentConfig(BaseModel):
    id: str
    model_id: str
    backend_runtime_id: str
    gpu_group_id: str | None = None
    gpu_group_ids: list[str] = Field(default_factory=list)
    api_model_name: str
    display_name: str
    memory_fraction: float = 0.9
    memory_fraction_overrides: dict[str, float] = Field(default_factory=dict)
    enabled: bool = True
    priority_weight: int = 100
    tasks: list[Literal["chat", "completion", "embedding"]] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def normalize_gpu_groups(self) -> "DeploymentConfig":
        normalized = list(self.gpu_group_ids)
        if self.gpu_group_id:
            if self.gpu_group_id in normalized:
                normalized = [self.gpu_group_id] + [item for item in normalized if item != self.gpu_group_id]
            else:
                normalized = [self.gpu_group_id, *normalized]
        deduped: list[str] = []
        for item in normalized:
            if item and item not in deduped:
                deduped.append(item)
        if not deduped:
            raise ValueError("Deployment requires gpu_group_id or gpu_group_ids")
        self.gpu_group_ids = deduped
        self.gpu_group_id = deduped[0]
        for gpu_group_id in self.memory_fraction_overrides:
            if gpu_group_id not in self.gpu_group_ids:
                raise ValueError(
                    f"memory_fraction_overrides contains unknown gpu group '{gpu_group_id}' "
                    f"for deployment '{self.id}'"
                )
        return self

    def eligible_gpu_group_ids(self) -> list[str]:
        return list(self.gpu_group_ids)

    def preferred_gpu_group_id(self) -> str:
        return self.gpu_group_ids[0]

    def memory_fraction_for_group(self, gpu_group_id: str) -> float:
        return self.memory_fraction_overrides.get(gpu_group_id, self.memory_fraction)


class GPUDeviceConfig(BaseModel):
    id: str
    uuid: str | None = None
    pci_bus_id: str | None = None
    vendor: str = "nvidia"
    observed_name: str | None = None
    cuda_visible_index: int | None = None


class GPUGroupConfig(BaseModel):
    id: str
    display_name: str
    gpu_ids: list[str]
    usable_memory_fraction: float = 0.95
    keep_free: bool = False
    scheduling_tags: list[str] = Field(default_factory=list)


class APIKeyRecord(BaseModel):
    id: str
    user_name: str
    priority: int = 100
    permissions: list[str] = Field(default_factory=lambda: ["inference"])
    secret_hash: str
    secret_prefix: str
    created_at: str
    revoked_at: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AdminUserRecord(BaseModel):
    username: str
    password_hash: str
    created_at: str
    disabled: bool = False

    @field_validator("created_at", mode="before")
    @classmethod
    def _normalize_created_at(cls, value: Any) -> str:
        if isinstance(value, datetime):
            return value.isoformat()
        return str(value)


class RepositoryConfig(BaseModel):
    global_config: GlobalConfig
    backends: dict[str, BackendRuntimeConfig]
    models: dict[str, ModelConfig]
    deployments: dict[str, DeploymentConfig]
    gpu_devices: dict[str, GPUDeviceConfig]
    gpu_groups: dict[str, GPUGroupConfig]
    admin_users: dict[str, AdminUserRecord]
    policies: dict[str, Any] = Field(default_factory=dict)
