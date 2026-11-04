from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


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


class ClearQueueRequest(BaseModel):
    interactive: bool = True
    batches: bool = False


class ChatFunctionCall(BaseModel):
    name: str
    arguments: str

    model_config = ConfigDict(extra="allow")


class ChatToolCall(BaseModel):
    """A tool call previously emitted by the assistant and replayed by the client."""

    id: str
    type: Literal["function"] = "function"
    function: ChatFunctionCall

    model_config = ConfigDict(extra="allow")


class ChatMessage(BaseModel):
    role: Literal["system", "developer", "user", "assistant", "tool"]
    # String, multimodal content parts, or null (assistant turns that only call tools).
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    tool_calls: list[ChatToolCall] | None = None
    tool_call_id: str | None = None

    # Backend-specific message fields (e.g. reasoning_content) pass through untouched.
    model_config = ConfigDict(extra="allow")


class ChatFunctionDefinition(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None
    strict: bool | None = None

    model_config = ConfigDict(extra="allow")


class ChatTool(BaseModel):
    type: Literal["function"] = "function"
    function: ChatFunctionDefinition

    model_config = ConfigDict(extra="allow")


class ChatNamedFunction(BaseModel):
    name: str


class ChatNamedToolChoice(BaseModel):
    type: Literal["function"] = "function"
    function: ChatNamedFunction


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float | None = None
    max_tokens: int | None = None
    response_format: dict[str, Any] | None = None
    reasoning_effort: Literal["none", "low", "medium", "high", "xhigh"] | None = None
    tools: list[ChatTool] | None = None
    tool_choice: Literal["none", "auto", "required"] | ChatNamedToolChoice | None = None
    parallel_tool_calls: bool | None = None
    stream: bool | None = None
    stream_options: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None

    # Forward every other OpenAI/vLLM sampling field (top_p, stop, seed, logprobs,
    # chat_template_kwargs, ...) instead of silently dropping it.
    model_config = ConfigDict(extra="allow")

    def backend_payload(self) -> dict[str, Any]:
        """Payload for the backend: exactly the fields the client sent.

        ``exclude_unset`` keeps nested defaults such as ``tool_calls: null`` out of
        replayed messages, which some chat templates treat as "defined".
        """
        return self.model_dump(mode="json", exclude_unset=True)


class ResponseCreateRequest(BaseModel):
    model: str
    input: Any | None = None

    model_config = ConfigDict(extra="allow")


class CompletionRequest(BaseModel):
    model: str
    prompt: str | list[str]
    max_tokens: int | None = None
    temperature: float | None = None
    stream: bool | None = None

    model_config = ConfigDict(extra="allow")


class EmbeddingRequest(BaseModel):
    model: str
    input: str | list[str]


class AudioSpeechRequest(BaseModel):
    model: str
    input: str
    voice: str
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] | None = None
    speed: float | None = None
    instructions: str | None = None


class AudioMultipartRequest(BaseModel):
    model: str
    language: str | None = None
    prompt: str | None = None
    response_format: Literal["json", "text", "srt", "verbose_json", "vtt"] | None = None
    temperature: float | None = None
    timestamp_granularities: list[Literal["word", "segment"]] = Field(default_factory=list)


class BatchCreateRequest(BaseModel):
    model: str
    requests: list[dict[str, Any]]
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelOnboardingRequest(BaseModel):
    model_id: str
    source: str
    display_name: str
    backend_compatibility: list[str]
    tasks: list[
        Literal[
            "chat",
            "response",
            "completion",
            "embedding",
            "audio_speech",
            "audio_transcription",
            "audio_translation",
        ]
    ] = Field(default_factory=list)
    model_capabilities: list[Literal["text", "audio", "image", "video"]] = Field(
        default_factory=lambda: ["text"]
    )
    runtime_launch_args: list[str] = Field(default_factory=list)
    runtime_launch_args_by_backend_type: dict[str, list[str]] = Field(default_factory=dict)
    tokenizer: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    create_deployment: bool = True
    deployment_id: str | None = None
    api_model_name: str | None = None
    deployment_display_name: str | None = None
    backend_runtime_id: str | None = None
    gpu_group_id: str | None = None
    gpu_group_ids: list[str] = Field(default_factory=list)
    memory_fraction: float = 0.9
    enabled: bool = True
    priority_weight: int = 100


class EnvironmentStatusResponse(BaseModel):
    hf_token_configured: bool
    hf_home: str | None = None
    environment_file: str | None = None


class RuntimeLoadRequest(BaseModel):
    deployment_id: str | None = None
    model_name: str | None = None
    gpu_group_id: str | None = None
