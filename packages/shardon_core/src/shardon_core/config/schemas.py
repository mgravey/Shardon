from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


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
    capabilities: BackendCapabilities = Field(default_factory=BackendCapabilities)


class ModelConfig(BaseModel):
    id: str
    source: str
    display_name: str
    backend_compatibility: list[str]
    tasks: list[Literal["chat", "completion", "embedding"]] = Field(default_factory=list)
    tokenizer: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DeploymentConfig(BaseModel):
    id: str
    model_id: str
    backend_runtime_id: str
    gpu_group_id: str
    api_model_name: str
    display_name: str
    memory_fraction: float = 0.9
    enabled: bool = True
    priority_weight: int = 100
    tasks: list[Literal["chat", "completion", "embedding"]] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)


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


class RepositoryConfig(BaseModel):
    global_config: GlobalConfig
    backends: dict[str, BackendRuntimeConfig]
    models: dict[str, ModelConfig]
    deployments: dict[str, DeploymentConfig]
    gpu_devices: dict[str, GPUDeviceConfig]
    gpu_groups: dict[str, GPUGroupConfig]
    admin_users: dict[str, AdminUserRecord]
    policies: dict[str, Any] = Field(default_factory=dict)
