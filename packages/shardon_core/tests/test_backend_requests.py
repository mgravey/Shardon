import asyncio
import json

import httpx
import pytest
from pydantic import ValidationError
from shardon_core.backends.base import (
    OpenAIHTTPBackendAdapter,
    backend_client_error,
    is_backend_connection_error,
)
from shardon_core.config.schemas import BackendRuntimeConfig, BackendRuntimeOverride


def _backend(**overrides) -> BackendRuntimeConfig:
    return BackendRuntimeConfig(
        id="backend-test",
        backend_type="mock",
        version="1",
        display_name="Test",
        runtime_dir=".",
        base_url="http://backend.test",
        launch_command=["python3"],
        **overrides,
    )


@pytest.mark.parametrize("custom", [False, True])
@pytest.mark.parametrize("operation", ["chat", "response", "completion", "embeddings"])
def test_json_requests_use_configured_timeouts(monkeypatch, custom, operation) -> None:
    backend = _backend()
    expected_read, expected_connect = 600.0, 10.0
    if custom:
        backend = _backend(
            request_read_timeout_seconds=240,
            request_connect_timeout_seconds=3,
            gpu_group_overrides={"group-b": {"request_read_timeout_seconds": 900}},
        ).resolved_for_gpu_group("group-b")
        expected_read, expected_connect = 900.0, 3.0

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.extensions["timeout"] == {
            "read": expected_read,
            "connect": expected_connect,
            "write": 120.0,
            "pool": 120.0,
        }
        assert json.loads(request.content) == {"model": "test"}
        return httpx.Response(200, json={"id": "ok"})

    client_type = httpx.AsyncClient
    monkeypatch.setattr(
        "shardon_core.backends.base.httpx.AsyncClient",
        lambda **kwargs: client_type(transport=httpx.MockTransport(handle), **kwargs),
    )
    adapter = OpenAIHTTPBackendAdapter(backend)
    result = asyncio.run(getattr(adapter, f"invoke_{operation}")({"model": "test", "unused": None}))
    assert result == {"id": "ok"}


def test_connect_timeout_can_be_overridden_independently() -> None:
    backend = _backend(
        request_read_timeout_seconds=250,
        request_connect_timeout_seconds=8,
        gpu_group_overrides={"group-b": {"request_connect_timeout_seconds": 2}},
    )
    resolved = backend.resolved_for_gpu_group("group-b")
    assert resolved.request_read_timeout_seconds == 250
    assert resolved.request_connect_timeout_seconds == 2
    assert backend.request_connect_timeout_seconds == 8


@pytest.mark.parametrize(
    "field", ["request_read_timeout_seconds", "request_connect_timeout_seconds"]
)
@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_timeouts_reject_nonpositive_or_nonfinite_values(field, value) -> None:
    with pytest.raises(ValidationError):
        _backend(**{field: value})
    with pytest.raises(ValidationError):
        BackendRuntimeOverride(**{field: value})


@pytest.mark.parametrize(
    ("error_type", "recover"),
    [
        (httpx.ReadTimeout, False),
        (httpx.WriteTimeout, False),
        (httpx.PoolTimeout, False),
        (httpx.ConnectTimeout, True),
        (httpx.ConnectError, True),
        (ConnectionError, True),
    ],
)
@pytest.mark.parametrize("wrapped", [False, True])
def test_backend_connection_error_classification(error_type, recover, wrapped) -> None:
    error = error_type("test")
    if wrapped:
        wrapper = RuntimeError("adapter failed")
        wrapper.__cause__ = error
        error = wrapper
    assert is_backend_connection_error(error) is recover


def _mock_transport(monkeypatch, handle) -> None:
    client_type = httpx.AsyncClient
    monkeypatch.setattr(
        "shardon_core.backends.base.httpx.AsyncClient",
        lambda **kwargs: client_type(transport=httpx.MockTransport(handle), **kwargs),
    )


def test_chat_stream_forwards_tool_payload_and_tool_call_deltas(monkeypatch) -> None:
    tools = [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]
    frames = [
        b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1",'
        b'"function":{"name":"lookup","arguments":""}}]}}]}\n\n',
        b'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n\n',
        b"data: [DONE]\n\n",
    ]

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        body = json.loads(request.content)
        assert body["tools"] == tools
        assert body["tool_choice"] == "auto"
        assert body["stream"] is True
        return httpx.Response(200, content=b"".join(frames))

    _mock_transport(monkeypatch, handle)
    adapter = OpenAIHTTPBackendAdapter(_backend())

    async def collect() -> bytes:
        payload = {"model": "m", "messages": [], "tools": tools, "tool_choice": "auto",
                   "stream": True}
        return b"".join([chunk async for chunk in adapter.stream_chat(payload)])

    body = asyncio.run(collect())
    assert b'"tool_calls"' in body
    assert b'"finish_reason":"tool_calls"' in body


@pytest.mark.parametrize("streaming", [False, True])
def test_backend_client_errors_keep_backend_body(monkeypatch, streaming) -> None:
    error = {"error": {"message": '"auto" tool choice requires --enable-auto-tool-choice'}}
    _mock_transport(monkeypatch, lambda request: httpx.Response(400, json=error))
    adapter = OpenAIHTTPBackendAdapter(_backend())

    async def call() -> None:
        if streaming:
            async for _ in adapter.stream_chat({"model": "m"}):
                pass
        else:
            await adapter.invoke_chat({"model": "m"})

    with pytest.raises(httpx.HTTPStatusError) as caught:
        asyncio.run(call())
    assert backend_client_error(caught.value) == (400, error)


def test_backend_server_errors_are_not_client_errors() -> None:
    request = httpx.Request("POST", "http://backend.test/v1/chat/completions")
    response = httpx.Response(500, json={"error": "boom"}, request=request)
    exc = httpx.HTTPStatusError("boom", request=request, response=response)
    assert backend_client_error(exc) is None
    assert backend_client_error(RuntimeError("x")) is None
