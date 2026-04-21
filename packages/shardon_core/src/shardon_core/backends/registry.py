from __future__ import annotations

import asyncio
from typing import Any

from pathlib import Path

from shardon_core.backends.base import BackendOperationError, OpenAIHTTPBackendAdapter, ProcessSupervisor
from shardon_core.config.schemas import BackendRuntimeConfig, DeploymentConfig, RepositoryConfig
from shardon_core.logging.events import EventLogger
from shardon_core.utils.time import utc_now_iso


class BackendRegistry:
    def __init__(self, config: RepositoryConfig, state_root: Path, event_logger: EventLogger) -> None:
        self.config = config
        self.event_logger = event_logger
        self.supervisor = ProcessSupervisor(state_root)

    def adapter_for(self, backend_runtime_id: str) -> OpenAIHTTPBackendAdapter:
        backend = self.config.backends[backend_runtime_id]
        return OpenAIHTTPBackendAdapter(backend)

    def ensure_started(self, deployment: DeploymentConfig) -> int:
        backend = self.config.backends[deployment.backend_runtime_id]
        model = self.config.models.get(deployment.model_id)
        managed = self.supervisor.start(
            backend=backend,
            deployment=deployment,
            extra_env={
                "SHARDON_DEPLOYMENT_ID": deployment.id,
                "SHARDON_MODEL_ID": deployment.model_id,
                "SHARDON_MODEL_SOURCE": model.source if model is not None else "",
                "SHARDON_MODEL_DISPLAY_NAME": model.display_name if model is not None else deployment.display_name,
                "SHARDON_API_MODEL_NAME": deployment.api_model_name,
                "SHARDON_GPU_GROUP_ID": deployment.gpu_group_id,
            },
        )
        self.event_logger.emit(
            "backend.start",
            "started backend runtime",
            deployment_id=deployment.id,
            backend_runtime_id=deployment.backend_runtime_id,
            pid=managed.pid,
        )
        return managed.pid

    async def ensure_started_and_ready(
        self,
        deployment: DeploymentConfig,
    ) -> dict[str, Any]:
        backend = self.config.backends[deployment.backend_runtime_id]
        timeout_seconds = (
            backend.startup_timeout_seconds
            or self.config.global_config.backend_startup_timeout_seconds
        )
        poll_interval_seconds = (
            backend.readiness_poll_interval_seconds
            or self.config.global_config.backend_readiness_poll_interval_seconds
        )
        pid = self.ensure_started(deployment)
        start_monotonic = asyncio.get_event_loop().time()
        last_error = "backend did not report readiness"
        attempt_count = 0
        self.event_logger.emit(
            "backend.readiness_wait_started",
            "waiting for backend readiness",
            deployment_id=deployment.id,
            backend_runtime_id=deployment.backend_runtime_id,
            pid=pid,
            timeout_seconds=timeout_seconds,
        )
        while asyncio.get_event_loop().time() - start_monotonic < timeout_seconds:
            attempt_count += 1
            if not self.supervisor.is_running(deployment.id):
                last_error = "backend process exited before readiness"
                break
            try:
                payload = await self.health(deployment.backend_runtime_id)
                detail = {
                    "deployment_id": deployment.id,
                    "backend_runtime_id": deployment.backend_runtime_id,
                    "pid": pid,
                    "attempt_count": attempt_count,
                    "ready_at": utc_now_iso(),
                    "payload": payload,
                }
                self.event_logger.emit(
                    "backend.ready",
                    "backend passed readiness",
                    deployment_id=deployment.id,
                    backend_runtime_id=deployment.backend_runtime_id,
                    pid=pid,
                    attempt_count=attempt_count,
                )
                return detail
            except Exception as exc:
                last_error = str(exc)
                await asyncio.sleep(poll_interval_seconds)
        self.stop(deployment.id, force=True)
        detail = {
            "deployment_id": deployment.id,
            "backend_runtime_id": deployment.backend_runtime_id,
            "pid": pid,
            "timeout_seconds": timeout_seconds,
            "attempt_count": attempt_count,
            "error": last_error,
        }
        self.event_logger.emit(
            "backend.readiness_failed",
            "backend failed readiness",
            **detail,
        )
        raise BackendOperationError("backend failed readiness", detail=detail)

    def stop(self, deployment_id: str, *, force: bool = False) -> None:
        backend = None
        if deployment_id in self.config.deployments:
            runtime_id = self.config.deployments[deployment_id].backend_runtime_id
            backend = self.config.backends.get(runtime_id)
        timeout = (
            backend.stop_timeout_seconds
            if backend is not None and backend.stop_timeout_seconds is not None
            else self.config.global_config.backend_stop_timeout_seconds
        )
        if force:
            self.supervisor.kill(deployment_id)
            self.event_logger.emit("backend.kill", "force killed backend runtime", deployment_id=deployment_id)
        else:
            self.supervisor.stop(deployment_id, timeout=timeout)
            self.event_logger.emit("backend.stop", "stopped backend runtime", deployment_id=deployment_id)

    async def health(self, backend_runtime_id: str) -> dict[str, Any]:
        adapter = self.adapter_for(backend_runtime_id)
        return await adapter.health()
