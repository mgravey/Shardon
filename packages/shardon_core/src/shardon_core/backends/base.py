from __future__ import annotations

import os
import signal
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from shardon_core.config.schemas import BackendRuntimeConfig, DeploymentConfig
from shardon_core.utils.time import utc_now_iso


class BackendOperationError(RuntimeError):
    def __init__(self, message: str, *, detail: dict[str, Any]) -> None:
        super().__init__(message)
        self.detail = detail


@dataclass(slots=True)
class ManagedProcess:
    deployment_id: str
    pid: int
    command: list[str]
    log_path: Path
    started_at: str


@dataclass(slots=True)
class UploadedFilePayload:
    filename: str
    content: bytes
    content_type: str | None = None


@dataclass(slots=True)
class BackendBinaryResponse:
    body: bytes
    content_type: str | None = None
    headers: dict[str, str] | None = None


class ProcessSupervisor:
    def __init__(self, state_root: Path) -> None:
        self.state_root = state_root
        self.processes: dict[str, ManagedProcess] = {}

    def start(
        self,
        *,
        backend: BackendRuntimeConfig,
        deployment: DeploymentConfig,
        extra_env: dict[str, str] | None = None,
    ) -> ManagedProcess:
        if deployment.id in self.processes:
            return self.processes[deployment.id]
        log_dir = self.state_root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{deployment.id}.log"
        env = os.environ.copy()
        env.update(backend.environment)
        env.update(extra_env or {})
        with log_path.open("ab") as handle:
            process = subprocess.Popen(
                backend.launch_command,
                cwd=backend.working_directory or backend.runtime_dir,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        managed = ManagedProcess(
            deployment_id=deployment.id,
            pid=process.pid,
            command=backend.launch_command,
            log_path=log_path,
            started_at=utc_now_iso(),
        )
        self.processes[deployment.id] = managed
        return managed

    def adopt(
        self,
        *,
        deployment_id: str,
        pid: int,
        command: list[str],
        log_path: Path | None = None,
        started_at: str | None = None,
    ) -> ManagedProcess:
        if log_path is None:
            log_dir = self.state_root / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"{deployment_id}.log"
        managed = ManagedProcess(
            deployment_id=deployment_id,
            pid=pid,
            command=command,
            log_path=log_path,
            started_at=started_at or utc_now_iso(),
        )
        self.processes[deployment_id] = managed
        return managed

    def stop(self, deployment_id: str, timeout: float = 10.0) -> None:
        managed = self.processes.get(deployment_id)
        if managed is None:
            return
        try:
            os.kill(managed.pid, signal.SIGTERM)
        except ProcessLookupError:
            self.processes.pop(deployment_id, None)
            return
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                os.kill(managed.pid, 0)
                time.sleep(0.1)
            except OSError:
                break
        else:
            try:
                os.kill(managed.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        self.processes.pop(deployment_id, None)

    def kill(self, deployment_id: str) -> None:
        managed = self.processes.get(deployment_id)
        if managed is None:
            return
        try:
            os.kill(managed.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self.processes.pop(deployment_id, None)

    def is_running(self, deployment_id: str) -> bool:
        managed = self.processes.get(deployment_id)
        if managed is None:
            return False
        try:
            os.kill(managed.pid, 0)
        except OSError:
            self.processes.pop(deployment_id, None)
            return False
        return True


class BackendAdapter(ABC):
    def __init__(self, backend: BackendRuntimeConfig) -> None:
        self.backend = backend

    async def health(self) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{self.backend.base_url}{self.backend.health_path}")
            response.raise_for_status()
            try:
                return response.json()
            except ValueError:
                payload: dict[str, Any] = {
                    "status": "ok",
                    "status_code": response.status_code,
                    "content_type": response.headers.get("content-type"),
                }
                body_text = response.text.strip()
                if body_text:
                    payload["body"] = body_text
                return payload

    @abstractmethod
    async def invoke_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    async def invoke_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    async def invoke_embeddings(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    async def invoke_audio_speech(self, payload: dict[str, Any]) -> BackendBinaryResponse:
        raise NotImplementedError

    @abstractmethod
    async def invoke_audio_transcription(
        self,
        payload: dict[str, Any],
        uploaded_file: UploadedFilePayload,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    async def invoke_audio_translation(
        self,
        payload: dict[str, Any],
        uploaded_file: UploadedFilePayload,
    ) -> dict[str, Any]:
        raise NotImplementedError

    async def invoke_multimodal_operation(
        self,
        operation: str,
        *,
        payload: dict[str, Any],
        uploaded_file: UploadedFilePayload | None = None,
    ) -> dict[str, Any] | BackendBinaryResponse:
        raise BackendOperationError(
            "backend does not implement multimodal operation",
            detail={"error": "unsupported multimodal operation", "operation": operation},
        )


class OpenAIHTTPBackendAdapter(BackendAdapter):
    def _clean_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in payload.items() if value is not None}

    async def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.backend.base_url}{path}",
                json=self._clean_payload(payload),
            )
            response.raise_for_status()
            return response.json()

    async def _post_json_binary(self, path: str, payload: dict[str, Any]) -> BackendBinaryResponse:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.backend.base_url}{path}",
                json=self._clean_payload(payload),
            )
            response.raise_for_status()
            return BackendBinaryResponse(
                body=response.content,
                content_type=response.headers.get("content-type"),
                headers=dict(response.headers),
            )

    def _multipart_data(self, payload: dict[str, Any]) -> list[tuple[str, str]]:
        data: list[tuple[str, str]] = []
        for key, value in self._clean_payload(payload).items():
            if isinstance(value, list):
                for item in value:
                    data.append((key, str(item)))
            else:
                data.append((key, str(value)))
        return data

    async def _post_multipart(
        self,
        path: str,
        *,
        payload: dict[str, Any],
        uploaded_file: UploadedFilePayload,
    ) -> dict[str, Any]:
        files = {
            "file": (
                uploaded_file.filename,
                uploaded_file.content,
                uploaded_file.content_type or "application/octet-stream",
            )
        }
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{self.backend.base_url}{path}",
                data=self._multipart_data(payload),
                files=files,
            )
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            if "application/json" in content_type:
                return response.json()
            return {"text": response.text}

    async def invoke_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._post_json("/v1/chat/completions", payload)

    async def invoke_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._post_json("/v1/completions", payload)

    async def invoke_embeddings(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._post_json("/v1/embeddings", payload)

    async def invoke_audio_speech(self, payload: dict[str, Any]) -> BackendBinaryResponse:
        return await self._post_json_binary("/v1/audio/speech", payload)

    async def invoke_audio_transcription(
        self,
        payload: dict[str, Any],
        uploaded_file: UploadedFilePayload,
    ) -> dict[str, Any]:
        return await self._post_multipart(
            "/v1/audio/transcriptions",
            payload=payload,
            uploaded_file=uploaded_file,
        )

    async def invoke_audio_translation(
        self,
        payload: dict[str, Any],
        uploaded_file: UploadedFilePayload,
    ) -> dict[str, Any]:
        return await self._post_multipart(
            "/v1/audio/translations",
            payload=payload,
            uploaded_file=uploaded_file,
        )


class WhisperXBackendAdapter(OpenAIHTTPBackendAdapter):
    def _path(self, key: str, default: str) -> str:
        return str(self.backend.capabilities.extra.get(key, default))

    def _normalize_whisperx_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        if "text" in payload:
            return payload
        segments = payload.get("segments")
        if isinstance(segments, list):
            text_segments = [
                str(item.get("text", "")).strip()
                for item in segments
                if isinstance(item, dict) and item.get("text")
            ]
            if text_segments:
                payload["text"] = " ".join(text_segments).strip()
        return payload

    async def invoke_audio_transcription(
        self,
        payload: dict[str, Any],
        uploaded_file: UploadedFilePayload,
    ) -> dict[str, Any]:
        data = await self._post_multipart(
            self._path("whisperx_transcriptions_path", "/asr"),
            payload=payload,
            uploaded_file=uploaded_file,
        )
        return self._normalize_whisperx_response(data)

    async def invoke_audio_translation(
        self,
        payload: dict[str, Any],
        uploaded_file: UploadedFilePayload,
    ) -> dict[str, Any]:
        data = await self._post_multipart(
            self._path("whisperx_translations_path", "/translate"),
            payload=payload,
            uploaded_file=uploaded_file,
        )
        return self._normalize_whisperx_response(data)
