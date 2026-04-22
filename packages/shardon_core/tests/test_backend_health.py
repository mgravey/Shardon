import asyncio
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from shardon_core.backends.base import BackendAdapter, BackendBinaryResponse, UploadedFilePayload
from shardon_core.config.schemas import BackendRuntimeConfig


class _DummyBackendAdapter(BackendAdapter):
    async def invoke_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        return payload

    async def invoke_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        return payload

    async def invoke_embeddings(self, payload: dict[str, Any]) -> dict[str, Any]:
        return payload

    async def invoke_audio_speech(self, payload: dict[str, Any]) -> BackendBinaryResponse:
        return BackendBinaryResponse(body=b"", content_type="audio/mpeg")

    async def invoke_audio_transcription(
        self,
        payload: dict[str, Any],
        uploaded_file: UploadedFilePayload,
    ) -> dict[str, Any]:
        _ = uploaded_file
        return payload

    async def invoke_audio_translation(
        self,
        payload: dict[str, Any],
        uploaded_file: UploadedFilePayload,
    ) -> dict[str, Any]:
        _ = uploaded_file
        return payload


def _make_backend() -> BackendRuntimeConfig:
    return BackendRuntimeConfig(
        id="backend-test",
        backend_type="vllm",
        version="1.0",
        display_name="backend-test",
        runtime_dir=".",
        base_url="http://backend.test",
        launch_command=["python3", "-m", "http.server"],
        health_path="/health",
    )


def _make_async_client_stub(
    response_factory: Callable[[str], httpx.Response],
) -> type:
    class _StubAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            _ = args
            _ = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            _ = exc_type
            _ = exc
            _ = tb

        async def get(self, url: str) -> httpx.Response:
            return response_factory(url)

    return _StubAsyncClient


def test_backend_health_returns_json_payload(monkeypatch) -> None:
    expected = {"ok": True}

    def response_factory(url: str) -> httpx.Response:
        return httpx.Response(200, json=expected, request=httpx.Request("GET", url))

    monkeypatch.setattr(
        "shardon_core.backends.base.httpx.AsyncClient",
        _make_async_client_stub(response_factory),
    )
    payload = asyncio.run(_DummyBackendAdapter(_make_backend()).health())
    assert payload == expected


def test_backend_health_accepts_plain_text_200(monkeypatch) -> None:
    def response_factory(url: str) -> httpx.Response:
        return httpx.Response(
            200,
            text="ok",
            headers={"content-type": "text/plain"},
            request=httpx.Request("GET", url),
        )

    monkeypatch.setattr(
        "shardon_core.backends.base.httpx.AsyncClient",
        _make_async_client_stub(response_factory),
    )
    payload = asyncio.run(_DummyBackendAdapter(_make_backend()).health())
    assert payload["status"] == "ok"
    assert payload["status_code"] == 200
    assert payload["content_type"] == "text/plain"
    assert payload["body"] == "ok"


def test_backend_health_raises_on_non_success_status(monkeypatch) -> None:
    def response_factory(url: str) -> httpx.Response:
        return httpx.Response(500, text="boom", request=httpx.Request("GET", url))

    monkeypatch.setattr(
        "shardon_core.backends.base.httpx.AsyncClient",
        _make_async_client_stub(response_factory),
    )
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(_DummyBackendAdapter(_make_backend()).health())
