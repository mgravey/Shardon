from __future__ import annotations

from pathlib import Path
from typing import Any

from shardon_core.backends.base import OpenAIHTTPBackendAdapter, ProcessSupervisor
from shardon_core.config.schemas import BackendRuntimeConfig, DeploymentConfig, RepositoryConfig
from shardon_core.logging.events import EventLogger


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

    def stop(self, deployment_id: str, *, force: bool = False) -> None:
        if force:
            self.supervisor.kill(deployment_id)
            self.event_logger.emit("backend.kill", "force killed backend runtime", deployment_id=deployment_id)
        else:
            self.supervisor.stop(deployment_id)
            self.event_logger.emit("backend.stop", "stopped backend runtime", deployment_id=deployment_id)

    async def health(self, backend_runtime_id: str) -> dict[str, Any]:
        adapter = self.adapter_for(backend_runtime_id)
        return await adapter.health()
