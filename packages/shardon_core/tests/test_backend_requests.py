import asyncio
import json

import httpx
import pytest
from pydantic import ValidationError
from shardon_core.backends.base import OpenAIHTTPBackendAdapter, is_backend_connection_error
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
