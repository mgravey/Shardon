import asyncio
from collections.abc import Callable
from typing import Literal

import httpx

from shardon_core.backends.base import OpenAIHTTPBackendAdapter, UploadedFilePayload, WhisperXBackendAdapter
from shardon_core.config.schemas import BackendRuntimeConfig


def _make_backend(
    *,
    backend_type: Literal["vllm", "sglang", "mock", "whisperx"] = "vllm",
    extra: dict | None = None,
) -> BackendRuntimeConfig:
    return BackendRuntimeConfig(
        id="backend-audio",
        backend_type=backend_type,
        version="1.0",
        display_name="backend-audio",
        runtime_dir=".",
        base_url="http://backend.test",
        launch_command=["python3", "-m", "http.server"],
        capabilities={
            "modalities": ["audio", "text"],
            "audio_speech": True,
            "audio_transcriptions": True,
            "audio_translations": True,
            "extra": extra or {},
        },
    )


def _make_async_client_stub(
    response_factory: Callable[..., httpx.Response],
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

        async def post(self, url: str, json=None, data=None, files=None) -> httpx.Response:  # type: ignore[no-untyped-def]
            return response_factory(url, json=json, data=data, files=files)

    return _StubAsyncClient


def test_audio_speech_returns_binary_payload(monkeypatch) -> None:
    def response_factory(url: str, *, json=None, data=None, files=None) -> httpx.Response:  # type: ignore[no-untyped-def]
        assert url.endswith("/v1/audio/speech")
        assert json["model"] == "audio-model"
        assert data is None
        assert files is None
        return httpx.Response(
            200,
            content=b"wave-bytes",
            headers={"content-type": "audio/wav"},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(
        "shardon_core.backends.base.httpx.AsyncClient",
        _make_async_client_stub(response_factory),
    )
    adapter = OpenAIHTTPBackendAdapter(_make_backend())
    payload = asyncio.run(adapter.invoke_audio_speech({"model": "audio-model", "input": "hello", "voice": "alloy"}))
    assert payload.body == b"wave-bytes"
    assert payload.content_type == "audio/wav"


def test_audio_transcription_uses_multipart_form(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def response_factory(url: str, *, json=None, data=None, files=None) -> httpx.Response:  # type: ignore[no-untyped-def]
        seen["url"] = url
        seen["json"] = json
        seen["data"] = data
        seen["files"] = files
        return httpx.Response(
            200,
            json={"text": "hello world"},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(
        "shardon_core.backends.base.httpx.AsyncClient",
        _make_async_client_stub(response_factory),
    )
    adapter = OpenAIHTTPBackendAdapter(_make_backend())
    result = asyncio.run(
        adapter.invoke_audio_transcription(
            {
                "model": "whisper-1",
                "timestamp_granularities": ["word", "segment"],
            },
            UploadedFilePayload(filename="speech.wav", content=b"abc", content_type="audio/wav"),
        )
    )
    assert result == {"text": "hello world"}
    assert str(seen["url"]).endswith("/v1/audio/transcriptions")
    assert seen["json"] is None
    assert ("model", "whisper-1") in (seen["data"] or [])
    assert ("timestamp_granularities", "word") in (seen["data"] or [])
    assert ("timestamp_granularities", "segment") in (seen["data"] or [])
    assert "file" in (seen["files"] or {})


def test_audio_translation_accepts_text_response(monkeypatch) -> None:
    def response_factory(url: str, *, json=None, data=None, files=None) -> httpx.Response:  # type: ignore[no-untyped-def]
        _ = json
        _ = data
        _ = files
        return httpx.Response(
            200,
            content=b"bonjour",
            headers={"content-type": "text/plain"},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(
        "shardon_core.backends.base.httpx.AsyncClient",
        _make_async_client_stub(response_factory),
    )
    adapter = OpenAIHTTPBackendAdapter(_make_backend())
    result = asyncio.run(
        adapter.invoke_audio_translation(
            {"model": "whisper-1"},
            UploadedFilePayload(filename="speech.wav", content=b"abc"),
        )
    )
    assert result == {"text": "bonjour"}


def test_whisperx_adapter_can_normalize_segments(monkeypatch) -> None:
    def response_factory(url: str, *, json=None, data=None, files=None) -> httpx.Response:  # type: ignore[no-untyped-def]
        _ = json
        _ = data
        _ = files
        assert url.endswith("/custom/asr")
        return httpx.Response(
            200,
            json={"segments": [{"text": "foo"}, {"text": "bar"}]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(
        "shardon_core.backends.base.httpx.AsyncClient",
        _make_async_client_stub(response_factory),
    )
    adapter = WhisperXBackendAdapter(
        _make_backend(
            backend_type="whisperx",
            extra={"whisperx_transcriptions_path": "/custom/asr"},
        )
    )
    result = asyncio.run(
        adapter.invoke_audio_transcription(
            {"model": "whisperx"},
            UploadedFilePayload(filename="speech.wav", content=b"abc"),
        )
    )
    assert result["text"] == "foo bar"
