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
            return response.json()

    @abstractmethod
    async def invoke_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    async def invoke_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    async def invoke_embeddings(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


class OpenAIHTTPBackendAdapter(BackendAdapter):
    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(f"{self.backend.base_url}{path}", json=payload)
            response.raise_for_status()
            return response.json()

    async def invoke_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._post("/v1/chat/completions", payload)

    async def invoke_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._post("/v1/completions", payload)

    async def invoke_embeddings(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._post("/v1/embeddings", payload)
