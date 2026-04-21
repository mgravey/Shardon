from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class AdminLoginRequest(BaseModel):
    username: str
    password: str


class AdminLoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class CreateAPIKeyRequest(BaseModel):
    key_id: str
    user_name: str
    priority: int = 100
    permissions: list[str] = Field(default_factory=lambda: ["inference"])
    metadata: dict[str, str] = Field(default_factory=dict)


class APIKeySecretResponse(BaseModel):
    id: str
    secret: str
    prefix: str


class DrainRequest(BaseModel):
    timeout_seconds: int = 120


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float | None = None
    max_tokens: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CompletionRequest(BaseModel):
    model: str
    prompt: str | list[str]
    max_tokens: int | None = None
    temperature: float | None = None


class EmbeddingRequest(BaseModel):
    model: str
    input: str | list[str]


class BatchCreateRequest(BaseModel):
    model: str
    requests: list[dict[str, Any]]
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelOnboardingRequest(BaseModel):
    model_id: str
    source: str
    display_name: str
    backend_compatibility: list[str]
    tasks: list[Literal["chat", "completion", "embedding"]] = Field(default_factory=list)
    tokenizer: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    create_deployment: bool = True
    deployment_id: str | None = None
    api_model_name: str | None = None
    deployment_display_name: str | None = None
    backend_runtime_id: str | None = None
    gpu_group_id: str | None = None
    memory_fraction: float = 0.9
    enabled: bool = True
    priority_weight: int = 100


class EnvironmentStatusResponse(BaseModel):
    hf_token_configured: bool
    hf_home: str | None = None
    environment_file: str | None = None
