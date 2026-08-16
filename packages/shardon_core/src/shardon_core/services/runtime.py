from __future__ import annotations

import asyncio
import getpass
import os
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Literal

from shardon_core.api.schemas import (
    AudioMultipartRequest,
    AudioSpeechRequest,
    BatchCreateRequest,
    ChatCompletionRequest,
    CompletionRequest,
    EmbeddingRequest,
    ResponseCreateRequest,
)
from shardon_core.auth.service import AdminAuthService, APIKeyService, AuthResult
from shardon_core.backends.base import (
    BackendBinaryResponse,
    BackendOperationError,
    UploadedFilePayload,
    is_backend_connection_error,
)
from shardon_core.backends.registry import BackendRegistry
from shardon_core.config.loader import load_repository_config
from shardon_core.config.schemas import DeploymentConfig, RepositoryConfig
from shardon_core.config.writer import delete_yaml, ensure_symlink, write_yaml
from shardon_core.gpu.provider import GPUProvider, NvidiaSMIProvider
from shardon_core.logging.events import EventLogger
from shardon_core.scheduler.engine import SchedulerEngine, SchedulingRequest
from shardon_core.state.models import (
    ActiveRequest,
    BatchJobState,
    DeploymentRuntimeState,
    DrainState,
    RuntimeStateSnapshot,
)
from shardon_core.state.store import RuntimeStateStore
from shardon_core.utils.time import utc_now, utc_now_iso


class RuntimeOperationError(RuntimeError):
    def __init__(self, message: str, *, detail: dict[str, Any], status_code: int = 409) -> None:
        super().__init__(message)
        self.detail = detail
        self.status_code = status_code


class ShardonRuntime:
    CURRENTLY_LOADED_MODEL_ALIAS = "currently-loaded"

    def __init__(
        self,
        *,
        repo_root: Path,
        gpu_provider: GPUProvider | None = None,
    ) -> None:
        self.repo_root = repo_root
        self.config_root = repo_root / "config"
        self.config: RepositoryConfig = load_repository_config(self.config_root)
        self.state_root = repo_root / self.config.global_config.state_root
        self.event_logger = EventLogger(self.state_root)
        self.state_store = RuntimeStateStore(self.state_root, self.event_logger)
        self.api_keys = APIKeyService(self.state_root, self.event_logger)
        self.admin_auth = AdminAuthService(
            admin_users=self.config.admin_users,
            state_root=self.state_root,
            event_logger=self.event_logger,
        )
        self.gpu_provider = gpu_provider or NvidiaSMIProvider()
        self.scheduler = SchedulerEngine(self.config)
        self.backends = BackendRegistry(self.config, self.state_root, self.event_logger)
        self._deployment_locks: dict[str, asyncio.Lock] = {}
        self.service_user = getpass.getuser()
        self.state_store.mutate(self._reconcile_loaded_processes)

    def _deployment_lock(self, deployment_id: str) -> asyncio.Lock:
        return self._deployment_locks.setdefault(deployment_id, asyncio.Lock())

    def reload_config(self) -> RepositoryConfig:
        self.config = load_repository_config(self.config_root)
        self.scheduler = SchedulerEngine(self.config)
        self.backends.config = self.config
        self.admin_auth.admin_users = self.config.admin_users
        return self.config

    def _config_paths(self, collection: str, item_id: str) -> tuple[Path, Path]:
        available = self.config_root / f"{collection}-available" / f"{item_id}.yaml"
        enabled = self.config_root / f"{collection}-enabled" / f"{item_id}.yaml"
        return available, enabled

    def upsert_config_item(
        self,
        *,
        collection: str,
        item_id: str,
        payload: dict[str, Any],
        enabled: bool = True,
    ) -> None:
        available, enabled_path = self._config_paths(collection, item_id)
        write_yaml(available, payload)
        if enabled:
            ensure_symlink(enabled_path, available)
        elif enabled_path.exists() or enabled_path.is_symlink():
            enabled_path.unlink()
        self.reload_config()

    def onboard_model(
        self,
        *,
        model_payload: dict[str, Any],
        deployment_payload: dict[str, Any] | None,
        actor: str,
    ) -> None:
        self.upsert_config_item(
            collection="models",
            item_id=model_payload["id"],
            payload=model_payload,
        )
        if deployment_payload is not None:
            self.upsert_config_item(
                collection="deployments",
                item_id=deployment_payload["id"],
                payload=deployment_payload,
            )
        self.event_logger.audit(
            "model.onboarded",
            actor,
            model_id=model_payload["id"],
            deployment_id=deployment_payload["id"] if deployment_payload is not None else None,
        )

    def delete_config_item(self, *, collection: str, item_id: str) -> None:
        available, enabled_path = self._config_paths(collection, item_id)
        delete_yaml(enabled_path)
        delete_yaml(available)
        self.reload_config()

    def read_backend_log(self, deployment_id: str, tail_lines: int = 200) -> list[str]:
        log_path = self.state_root / "logs" / f"{deployment_id}.log"
        if not log_path.exists():
            return []
        lines = log_path.read_text(encoding="utf-8").splitlines()
        return lines[-tail_lines:]

    def read_events(self, tail_lines: int = 200) -> list[str]:
        path = self.state_root / "events" / "events.jsonl"
        if not path.exists():
            return []
        return path.read_text(encoding="utf-8").splitlines()[-tail_lines:]

    def environment_status(self) -> dict[str, Any]:
        env_file = self.repo_root / ".env"
        return {
            "hf_token_configured": bool(os.environ.get("HF_TOKEN")),
            "hf_home": os.environ.get("HF_HOME"),
            "environment_file": str(env_file) if env_file.exists() else None,
        }

    def snapshot(self) -> RuntimeStateSnapshot:
        return self.state_store.load()

    def _seed_snapshot(self, snapshot: RuntimeStateSnapshot) -> RuntimeStateSnapshot:
        for deployment in self.config.deployments.values():
            runtime = snapshot.deployments.setdefault(
                deployment.id,
                DeploymentRuntimeState(
                    deployment_id=deployment.id,
                    gpu_group_id=deployment.preferred_gpu_group_id(),
                    eligible_gpu_group_ids=deployment.eligible_gpu_group_ids(),
                    selected_gpu_group_id=None,
                    backend_runtime_id=deployment.backend_runtime_id,
                ),
            )
            runtime.backend_runtime_id = deployment.backend_runtime_id
            runtime.eligible_gpu_group_ids = deployment.eligible_gpu_group_ids()
            if runtime.gpu_group_id not in runtime.eligible_gpu_group_ids:
                runtime.gpu_group_id = deployment.preferred_gpu_group_id()
            if runtime.selected_gpu_group_id not in runtime.eligible_gpu_group_ids:
                runtime.selected_gpu_group_id = None
            if runtime.loaded and runtime.selected_gpu_group_id is None:
                runtime.selected_gpu_group_id = runtime.gpu_group_id
        return snapshot

    def refresh_gpu_observations(self) -> RuntimeStateSnapshot:
        observations = self.gpu_provider.observe(self.config.gpu_devices)
        managed_pids = {process.pid for process in self.backends.supervisor.processes.values()}

        def mutate(snapshot: RuntimeStateSnapshot) -> RuntimeStateSnapshot:
            snapshot = self._seed_snapshot(snapshot)
            for gpu_id, observation in observations.items():
                snapshot.gpu_observations[gpu_id] = observation.model_copy(
                    update={
                        "observed_processes": [
                            process.model_copy(
                                update={"managed_by_shardon": process.pid in managed_pids}
                            )
                            for process in observation.observed_processes
                        ]
                    }
                )
            return snapshot

        return self.state_store.mutate(mutate)

    def enforce_keep_free(self) -> RuntimeStateSnapshot:
        def mutate(snapshot: RuntimeStateSnapshot) -> RuntimeStateSnapshot:
            snapshot = self._seed_snapshot(snapshot)
            for group in self.config.gpu_groups.values():
                if not group.keep_free:
                    continue
                observations = [snapshot.gpu_observations.get(gpu_id) for gpu_id in group.gpu_ids]
                external_users = {
                    process.user_name
                    for observation in observations
                    if observation is not None
                    for process in observation.observed_processes
                    if not process.managed_by_shardon and process.user_name != self.service_user
                }
                if not external_users:
                    continue
                for deployment_id, state in snapshot.deployments.items():
                    if state.gpu_group_id != group.id or not state.loaded:
                        continue
                    self.backends.stop(
                        deployment_id,
                        gpu_group_id=state.selected_gpu_group_id or state.gpu_group_id,
                        force=True,
                    )
                    state.loaded = False
                    state.state = "unloaded"
                    state.desired_state = "unloaded"
                    state.process_id = None
                    state.keep_free_killed_at = utc_now_iso()
                    state.active_request_ids = []
                    state.last_transition_reason = "keep_free enforcement"
                    self.event_logger.emit(
                        "keep_free.kill",
                        "killed runtime because another user was active on a keep-free group",
                        gpu_group_id=group.id,
                        deployment_id=deployment_id,
                        external_users=sorted(external_users),
                    )
            return snapshot

        return self.state_store.mutate(mutate)

    def list_api_models(self) -> list[dict[str, Any]]:
        snapshot = self.snapshot()
        models: dict[str, dict[str, Any]] = {}
        for deployment in self.config.deployments.values():
            if not deployment.enabled:
                continue
            model = self.config.models[deployment.model_id]
            backend = self.config.backends.get(deployment.backend_runtime_id)
            runtime = snapshot.deployments.get(deployment.id)
            models.setdefault(
                deployment.api_model_name,
                {
                    "id": deployment.api_model_name,
                    "object": "model",
                    "owned_by": "shardon",
                    "display_name": deployment.display_name,
                    "source_model_id": model.id,
                    "backend_runtime_id": deployment.backend_runtime_id,
                    "gpu_group_id": deployment.preferred_gpu_group_id(),
                    "gpu_group_ids": deployment.eligible_gpu_group_ids(),
                    "selected_gpu_group_id": runtime.selected_gpu_group_id if runtime is not None else None,
                    "tasks": deployment.tasks,
                    "model_capabilities": model.model_capabilities,
                    "deployment_capabilities": deployment.deployment_capabilities,
                    "backend_modalities": backend.capabilities.modalities if backend is not None else [],
                    "effective_capabilities": self._deployment_effective_capabilities(deployment),
                },
            )
        loaded = self._currently_loaded_deployments(snapshot=snapshot)
        if loaded and self.CURRENTLY_LOADED_MODEL_ALIAS not in models:
            deployment, runtime = loaded[0]
            model = self.config.models[deployment.model_id]
            backend = self.config.backends.get(deployment.backend_runtime_id)
            selected_gpu_group_id = runtime.selected_gpu_group_id or runtime.gpu_group_id
            models[self.CURRENTLY_LOADED_MODEL_ALIAS] = {
                "id": self.CURRENTLY_LOADED_MODEL_ALIAS,
                "object": "model",
                "owned_by": "shardon",
                "display_name": "Currently loaded",
                "source_model_id": model.id,
                "current_model_name": runtime.current_model_name or deployment.api_model_name,
                "resolved_model_id": deployment.api_model_name,
                "resolved_deployment_id": deployment.id,
                "backend_runtime_id": deployment.backend_runtime_id,
                "gpu_group_id": deployment.preferred_gpu_group_id(),
                "gpu_group_ids": deployment.eligible_gpu_group_ids(),
                "selected_gpu_group_id": selected_gpu_group_id,
                "tasks": deployment.tasks,
                "model_capabilities": model.model_capabilities,
                "deployment_capabilities": deployment.deployment_capabilities,
                "backend_modalities": backend.capabilities.modalities if backend is not None else [],
                "effective_capabilities": self._deployment_effective_capabilities(deployment),
            }
        return list(models.values())

    @staticmethod
    def _deployment_supports_task(deployment: DeploymentConfig, task: str) -> bool:
        if task in deployment.tasks:
            return True
        return task == "response" and "chat" in deployment.tasks

    def _currently_loaded_deployments(
        self,
        *,
        snapshot: RuntimeStateSnapshot,
        task: str | None = None,
        required_capability: Literal["text", "audio", "image", "video"] | None = None,
    ) -> list[tuple[DeploymentConfig, DeploymentRuntimeState]]:
        loaded: list[tuple[DeploymentConfig, DeploymentRuntimeState]] = []
        for deployment_id, runtime in snapshot.deployments.items():
            if not runtime.loaded or runtime.state != "ready":
                continue
            deployment = self.config.deployments.get(deployment_id)
            if deployment is None or not deployment.enabled:
                continue
            if task is not None and not self._deployment_supports_task(deployment, task):
                continue
            if (
                required_capability is not None
                and required_capability not in self._deployment_effective_capabilities(deployment)
            ):
                continue
            loaded.append((deployment, runtime))
        return sorted(
            loaded,
            key=lambda item: (item[1].last_used_at or item[1].loaded_at or "", item[0].id),
            reverse=True,
        )

    def _resolve_model_alias(
        self,
        *,
        model_name: str,
        task: str,
        required_capability: Literal["text", "audio", "image", "video"],
        snapshot: RuntimeStateSnapshot,
    ) -> str:
        if model_name != self.CURRENTLY_LOADED_MODEL_ALIAS:
            return model_name
        loaded = self._currently_loaded_deployments(
            snapshot=snapshot,
            task=task,
            required_capability=required_capability,
        )
        if not loaded:
            raise RuntimeOperationError(
                "no currently loaded deployment",
                detail={
                    "error": "no currently loaded deployment",
                    "model_name": model_name,
                    "task": task,
                    "required_capability": required_capability,
                },
                status_code=404,
            )
        deployment, _ = loaded[0]
        return deployment.api_model_name

    async def refresh_backend_health(self) -> RuntimeStateSnapshot:
        results: dict[str, dict[str, Any]] = {}
        snapshot = self.state_store.mutate(self._reconcile_loaded_processes)
        for deployment in self.config.deployments.values():
            runtime_state = snapshot.deployments.get(deployment.id)
            selected_gpu_group_id = (
                runtime_state.selected_gpu_group_id or runtime_state.gpu_group_id
                if runtime_state is not None
                else deployment.preferred_gpu_group_id()
            )
            if runtime_state is None or not (runtime_state.loaded or runtime_state.state == "starting"):
                results[deployment.id] = {
                    "ok": False,
                    "status": "not_loaded",
                    "checked_at": utc_now_iso(),
                    "deployment_id": deployment.id,
                    "backend_runtime_id": deployment.backend_runtime_id,
                    "gpu_group_id": selected_gpu_group_id,
                }
                continue
            try:
                payload = await self.backends.health(
                    deployment.backend_runtime_id,
                    gpu_group_id=selected_gpu_group_id,
                )
                results[deployment.id] = {
                    "ok": True,
                    "status": "ready",
                    "checked_at": utc_now_iso(),
                    "deployment_id": deployment.id,
                    "backend_runtime_id": deployment.backend_runtime_id,
                    "gpu_group_id": selected_gpu_group_id,
                    "payload": payload,
                }
            except Exception as exc:
                reason = f"backend health unreachable: {exc}"
                results[deployment.id] = {
                    "ok": False,
                    "status": "unreachable",
                    "checked_at": utc_now_iso(),
                    "deployment_id": deployment.id,
                    "backend_runtime_id": deployment.backend_runtime_id,
                    "gpu_group_id": selected_gpu_group_id,
                    "error": str(exc),
                }
                previous = snapshot.backend_health.get(deployment.id, {})
                if previous.get("status") != "unreachable" or previous.get("error") != str(exc):
                    self.event_logger.emit(
                        "backend.health_failed",
                        "backend health check failed",
                        deployment_id=deployment.id,
                        backend_runtime_id=deployment.backend_runtime_id,
                        gpu_group_id=selected_gpu_group_id,
                        old_pid=runtime_state.process_id,
                        reason=reason,
                        error=str(exc),
                    )
                if runtime_state.loaded:
                    try:
                        await self._invalidate_unhealthy_backend(
                            deployment,
                            selected_gpu_group_id=selected_gpu_group_id,
                            observed_pid=runtime_state.process_id,
                            reason=reason,
                            source="health_check",
                        )
                    except (asyncio.CancelledError, GeneratorExit):
                        raise
                    except Exception as cleanup_exc:
                        self.event_logger.emit(
                            "backend.recovery",
                            "failed to invalidate unhealthy backend",
                            deployment_id=deployment.id,
                            backend_runtime_id=deployment.backend_runtime_id,
                            gpu_group_id=selected_gpu_group_id,
                            old_pid=runtime_state.process_id,
                            reason=reason,
                            recovery_result="cleanup_failed",
                            error=str(cleanup_exc),
                        )
        return self.state_store.mutate(lambda snapshot: self._set_backend_health(snapshot, results))

    def resolve_deployment(
        self,
        *,
        deployment_id: str | None = None,
        model_name: str | None = None,
        gpu_group_id: str | None = None,
    ) -> DeploymentConfig:
        if deployment_id is not None:
            deployment = self.config.deployments.get(deployment_id)
            if deployment is None:
                raise RuntimeOperationError(
                    "unknown deployment",
                    detail={"error": "unknown deployment", "deployment_id": deployment_id},
                    status_code=404,
                )
            return deployment
        if model_name is None:
            raise RuntimeOperationError(
                "missing deployment selector",
                detail={"error": "provide deployment_id or model_name"},
                status_code=422,
            )
        matches = [
            deployment
            for deployment in self.config.deployments.values()
            if deployment.api_model_name == model_name
        ]
        if gpu_group_id is not None:
            matches = [
                deployment
                for deployment in matches
                if gpu_group_id in deployment.eligible_gpu_group_ids()
            ]
        if not matches:
            raise RuntimeOperationError(
                "no matching deployment",
                detail={"error": "no matching deployment", "model_name": model_name, "gpu_group_id": gpu_group_id},
                status_code=404,
            )
        if len(matches) > 1 and gpu_group_id is None:
            raise RuntimeOperationError(
                "ambiguous model selector",
                detail={
                    "error": "multiple deployments match model_name; provide deployment_id or gpu_group_id",
                    "model_name": model_name,
                    "deployment_ids": [deployment.id for deployment in matches],
                },
                status_code=409,
            )
        return matches[0]

    async def load_deployment(
        self,
        *,
        deployment_id: str | None = None,
        model_name: str | None = None,
        gpu_group_id: str | None = None,
        actor: str = "operator",
    ) -> dict[str, Any]:
        deployment = self.resolve_deployment(
            deployment_id=deployment_id,
            model_name=model_name,
            gpu_group_id=gpu_group_id,
        )
        if gpu_group_id is not None and gpu_group_id not in deployment.eligible_gpu_group_ids():
            raise RuntimeOperationError(
                "deployment cannot run on requested gpu group",
                detail={
                    "error": "deployment cannot run on requested gpu group",
                    "deployment_id": deployment.id,
                    "gpu_group_id": gpu_group_id,
                    "eligible_gpu_group_ids": deployment.eligible_gpu_group_ids(),
                },
                status_code=409,
            )
        snapshot = self.snapshot()
        runtime_state = snapshot.deployments.get(deployment.id)
        selected_gpu_group_id = (
            (runtime_state.selected_gpu_group_id or runtime_state.gpu_group_id)
            if runtime_state is not None
            else None
        )
        if runtime_state is not None and runtime_state.loaded and (
            gpu_group_id is None or gpu_group_id == selected_gpu_group_id
        ):
            return {
                "status": "ok",
                "detail": "deployment already loaded",
                "deployment_id": deployment.id,
                "gpu_group_id": selected_gpu_group_id,
                "selected_gpu_group_id": selected_gpu_group_id,
            }
        decision = self.scheduler.schedule(
            SchedulingRequest(
                model_name=deployment.api_model_name,
                task=deployment.tasks[0] if deployment.tasks else "chat",
                priority=10_000,
                request_class="manual",
                request_id=f"manual_load_{deployment.id}",
                deployment_id=deployment.id,
                target_gpu_group_id=gpu_group_id,
            ),
            snapshot,
            utc_now(),
        )
        if not decision.accepted or decision.deployment_id is None or decision.gpu_group_id is None:
            raise RuntimeOperationError(
                "deployment is not currently admissible",
                detail={
                    "error": "deployment is not currently admissible",
                    "deployment_id": deployment.id,
                    "reason": decision.reason,
                    **self._build_candidate_status(
                        model_name=deployment.api_model_name,
                        task=deployment.tasks[0] if deployment.tasks else "chat",
                        snapshot=snapshot,
                    ),
                },
                status_code=409,
            )
        if gpu_group_id is not None and decision.gpu_group_id != gpu_group_id:
            raise RuntimeOperationError(
                "requested gpu group is not currently admissible",
                detail={
                    "error": "requested gpu group is not currently admissible",
                    "deployment_id": deployment.id,
                    "requested_gpu_group_id": gpu_group_id,
                    "selected_gpu_group_id": decision.gpu_group_id,
                },
                status_code=409,
            )
        evict_ids = decision.should_evict or []
        self._prepare_group_for_load(evict_ids, reason=f"manual load requested by {actor}")
        await self._start_and_mark_ready(
            deployment,
            selected_gpu_group_id=decision.gpu_group_id,
            reason=f"manual load requested by {actor}",
        )
        self.event_logger.audit(
            "runtime.load",
            actor,
            deployment_id=deployment.id,
            gpu_group_id=decision.gpu_group_id,
        )
        return {
            "status": "ok",
            "deployment_id": deployment.id,
            "gpu_group_id": decision.gpu_group_id,
            "selected_gpu_group_id": decision.gpu_group_id,
        }

    async def unload_deployment(self, deployment_id: str, *, actor: str = "operator") -> dict[str, Any]:
        deployment = self.resolve_deployment(deployment_id=deployment_id)
        async with self._deployment_lock(deployment.id):
            snapshot = self.snapshot()
            runtime_state = snapshot.deployments.get(deployment.id)
            if runtime_state is None or not runtime_state.loaded:
                return {
                    "status": "ok",
                    "detail": "deployment already unloaded",
                    "deployment_id": deployment.id,
                }
            if runtime_state.active_request_ids:
                raise RuntimeOperationError(
                    "deployment still serving requests",
                    detail={
                        "error": "deployment still serving requests",
                        "deployment_id": deployment.id,
                        "active_request_ids": runtime_state.active_request_ids,
                    },
                    status_code=409,
                )
            self._ensure_supervised_process(deployment.id, runtime_state)
            self.backends.stop(
                deployment.id,
                gpu_group_id=runtime_state.selected_gpu_group_id or runtime_state.gpu_group_id,
                force=False,
            )
            self.state_store.mutate(
                lambda state: self._mark_unloaded(
                    state,
                    deployment.id,
                    reason=f"manual unload requested by {actor}",
                )
            )
        self.event_logger.audit("runtime.unload", actor, deployment_id=deployment.id)
        return {"status": "ok", "deployment_id": deployment.id}

    def clear_queue(
        self,
        *,
        clear_interactive: bool = True,
        clear_batches: bool = False,
        actor: str = "operator",
    ) -> dict[str, Any]:
        cleared_interactive_request_ids: list[str] = []
        cancelled_batch_ids: list[str] = []

        def mutate(snapshot: RuntimeStateSnapshot) -> RuntimeStateSnapshot:
            if clear_interactive:
                cleared_interactive_request_ids.extend([item.id for item in snapshot.queued_requests])
                snapshot.queued_requests = []
            if clear_batches:
                for job in snapshot.batch_jobs.values():
                    if job.status != "queued":
                        continue
                    job.status = "cancelled"
                    job.updated_at = utc_now_iso()
                    cancelled_batch_ids.append(job.id)
            return snapshot

        self.state_store.mutate(mutate)
        detail = {
            "status": "ok",
            "cleared_interactive_requests": len(cleared_interactive_request_ids),
            "cancelled_batch_jobs": len(cancelled_batch_ids),
            "interactive_request_ids": cleared_interactive_request_ids,
            "batch_job_ids": cancelled_batch_ids,
        }
        self.event_logger.audit("runtime.queue.cleared", actor, **detail)
        return detail

    async def route_chat(
        self,
        request: ChatCompletionRequest,
        auth: AuthResult,
    ) -> dict[str, Any]:
        return await self._route_interactive(
            "chat",
            request.model,
            request.model_dump(mode="json"),
            auth,
            required_capability="text",
        )

    async def route_completion(
        self,
        request: CompletionRequest,
        auth: AuthResult,
    ) -> dict[str, Any]:
        return await self._route_interactive(
            "completion",
            request.model,
            request.model_dump(mode="json"),
            auth,
            required_capability="text",
        )

    async def stream_completion(
        self,
        request: CompletionRequest,
        auth: AuthResult,
    ) -> AsyncIterator[bytes]:
        payload = request.model_dump(mode="json")
        async for chunk in self._route_interactive_stream(
            "completion",
            request.model,
            payload,
            auth,
            required_capability="text",
        ):
            yield chunk

    async def route_response(
        self,
        request: ResponseCreateRequest,
        auth: AuthResult,
    ) -> dict[str, Any]:
        return await self._route_interactive(
            "response",
            request.model,
            request.model_dump(mode="json"),
            auth,
            required_capability="text",
        )

    async def stream_response(
        self,
        request: ResponseCreateRequest,
        auth: AuthResult,
    ) -> AsyncIterator[bytes]:
        payload = request.model_dump(mode="json")
        async for chunk in self._route_interactive_stream(
            "response",
            request.model,
            payload,
            auth,
            required_capability="text",
        ):
            yield chunk

    async def route_embedding(
        self,
        request: EmbeddingRequest,
        auth: AuthResult,
    ) -> dict[str, Any]:
        return await self._route_interactive(
            "embedding",
            request.model,
            request.model_dump(mode="json"),
            auth,
            required_capability="text",
        )

    async def route_audio_speech(
        self,
        request: AudioSpeechRequest,
        auth: AuthResult,
    ) -> BackendBinaryResponse:
        response = await self._route_interactive(
            "audio_speech",
            request.model,
            request.model_dump(mode="json"),
            auth,
            required_capability="audio",
        )
        if not isinstance(response, BackendBinaryResponse):
            raise RuntimeOperationError(
                "backend returned non-binary response for audio speech",
                detail={"error": "invalid audio speech response"},
                status_code=409,
            )
        return response

    async def route_audio_transcription(
        self,
        request: AudioMultipartRequest,
        uploaded_file: UploadedFilePayload,
        auth: AuthResult,
    ) -> dict[str, Any]:
        response = await self._route_interactive(
            "audio_transcription",
            request.model,
            request.model_dump(mode="json"),
            auth,
            required_capability="audio",
            uploaded_file=uploaded_file,
        )
        if not isinstance(response, dict):
            raise RuntimeOperationError(
                "backend returned non-json response for transcription",
                detail={"error": "invalid transcription response"},
                status_code=409,
            )
        return response

    async def route_audio_translation(
        self,
        request: AudioMultipartRequest,
        uploaded_file: UploadedFilePayload,
        auth: AuthResult,
    ) -> dict[str, Any]:
        response = await self._route_interactive(
            "audio_translation",
            request.model,
            request.model_dump(mode="json"),
            auth,
            required_capability="audio",
            uploaded_file=uploaded_file,
        )
        if not isinstance(response, dict):
            raise RuntimeOperationError(
                "backend returned non-json response for translation",
                detail={"error": "invalid translation response"},
                status_code=409,
            )
        return response

    async def _route_interactive(
        self,
        task: str,
        model_name: str,
        payload: dict[str, Any],
        auth: AuthResult,
        *,
        required_capability: Literal["text", "audio", "image", "video"],
        uploaded_file: UploadedFilePayload | None = None,
    ) -> Any:
        self.refresh_gpu_observations()
        self.enforce_keep_free()
        request_id = f"req_{uuid.uuid4().hex}"
        preflight_snapshot = self.snapshot()
        requested_model_name = model_name
        model_name = self._resolve_model_alias(
            model_name=model_name,
            task=task,
            required_capability=required_capability,
            snapshot=preflight_snapshot,
        )
        if model_name != requested_model_name:
            payload = {**payload, "model": model_name}
        preflight_decision = self.scheduler.schedule(
            SchedulingRequest(
                model_name=model_name,
                task=task,
                priority=auth.priority,
                request_class="interactive",
                request_id=request_id,
                required_capability=required_capability,
            ),
            preflight_snapshot,
            utc_now(),
        )
        if not preflight_decision.accepted and preflight_decision.status_code == 404:
            detail = {
                "error": "no compatible deployment",
                "model_name": model_name,
                "task": task,
                "required_capability": required_capability,
                "last_decision_reason": preflight_decision.reason,
                **self._build_candidate_status(
                    model_name=model_name,
                    task=task,
                    required_capability=required_capability,
                    snapshot=preflight_snapshot,
                ),
            }
            raise RuntimeOperationError(
                "no compatible deployment",
                detail=detail,
                status_code=404,
            )
        queued_request = ActiveRequest(
            id=request_id,
            user_name=auth.user_name,
            api_key_id=auth.id,
            deployment_id="",
            backend_runtime_id="",
            gpu_group_id="",
            request_class="interactive",
            model_name=model_name,
            status="queued",
            priority=auth.priority,
            created_at=utc_now_iso(),
        )
        self.state_store.mutate(lambda snapshot: self._enqueue_request(snapshot, queued_request))
        interactive_timeout_seconds = (
            self.config.global_config.effective_interactive_request_timeout_seconds()
        )
        deadline = asyncio.get_event_loop().time() + interactive_timeout_seconds
        last_decision_reason = "no scheduling decision made yet"
        last_detail: dict[str, Any] = {}
        while asyncio.get_event_loop().time() < deadline:
            snapshot = self.snapshot()
            if not any(item.id == request_id for item in snapshot.queued_requests):
                detail = {
                    "error": "request cancelled",
                    "request_id": request_id,
                    "model_name": model_name,
                    "task": task,
                }
                raise RuntimeOperationError("request cancelled", detail=detail, status_code=409)
            decision = self.scheduler.schedule(
                SchedulingRequest(
                    model_name=model_name,
                    task=task,
                    priority=auth.priority,
                    request_class="interactive",
                    request_id=request_id,
                    required_capability=required_capability,
                ),
                snapshot,
                utc_now(),
            )
            last_decision_reason = decision.reason
            last_detail = self._build_candidate_status(
                model_name=model_name,
                task=task,
                required_capability=required_capability,
                snapshot=snapshot,
            )
            if not decision.accepted and decision.status_code == 404:
                self.state_store.mutate(lambda state: self._drop_request(state, request_id))
                detail = {
                    "error": "no compatible deployment",
                    "model_name": model_name,
                    "task": task,
                    "required_capability": required_capability,
                    "last_decision_reason": decision.reason,
                    **last_detail,
                }
                raise RuntimeOperationError(
                    "no compatible deployment",
                    detail=detail,
                    status_code=404,
                )
            if decision.accepted and decision.deployment_id is not None:
                if any(
                    snapshot.deployments.get(deployment_id)
                    and snapshot.deployments[deployment_id].active_request_ids
                    for deployment_id in (decision.should_evict or [])
                ):
                    await asyncio.sleep(self.config.global_config.queue_poll_interval_seconds)
                    continue
                deployment = self.config.deployments[decision.deployment_id]
                try:
                    response = await self._execute_request(
                        deployment=deployment,
                        request_id=request_id,
                        payload=payload,
                        task=task,
                        auth=auth,
                        target_gpu_group_id=(
                            decision.gpu_group_id or deployment.preferred_gpu_group_id()
                        ),
                        should_evict=decision.should_evict or [],
                        uploaded_file=uploaded_file,
                    )
                    self.state_store.mutate(lambda state: self._dequeue_request(state, request_id))
                    return response
                except RuntimeOperationError as exc:
                    diagnostic = exc.detail
                    self.state_store.mutate(
                        lambda state: self._mark_request_diagnostic(
                            state,
                            request_id,
                            diagnostic,
                        )
                    )
                    self.state_store.mutate(lambda state: self._drop_request(state, request_id))
                    raise
                except (asyncio.CancelledError, GeneratorExit):
                    self.state_store.mutate(lambda state: self._drop_request(state, request_id))
                    raise
            await asyncio.sleep(self.config.global_config.queue_poll_interval_seconds)
            self.refresh_gpu_observations()
            self.enforce_keep_free()
        self.state_store.mutate(lambda state: self._drop_request(state, request_id))
        detail = {
            "error": "request timed out waiting for a deployment",
            "model_name": model_name,
            "task": task,
            "timeout_seconds": interactive_timeout_seconds,
            "last_decision_reason": last_decision_reason,
            **last_detail,
        }
        self.event_logger.emit(
            "routing.timeout",
            "request timed out waiting for a deployment",
            request_id=request_id,
            **detail,
        )
        raise RuntimeOperationError(
            "request timed out waiting for a deployment",
            detail=detail,
            status_code=409,
        )

    async def _route_interactive_stream(
        self,
        task: str,
        model_name: str,
        payload: dict[str, Any],
        auth: AuthResult,
        *,
        required_capability: Literal["text", "audio", "image", "video"],
    ) -> AsyncIterator[bytes]:
        self.refresh_gpu_observations()
        self.enforce_keep_free()
        request_id = f"req_{uuid.uuid4().hex}"
        preflight_snapshot = self.snapshot()
        requested_model_name = model_name
        model_name = self._resolve_model_alias(
            model_name=model_name,
            task=task,
            required_capability=required_capability,
            snapshot=preflight_snapshot,
        )
        if model_name != requested_model_name:
            payload = {**payload, "model": model_name}
        preflight_decision = self.scheduler.schedule(
            SchedulingRequest(
                model_name=model_name,
                task=task,
                priority=auth.priority,
                request_class="interactive",
                request_id=request_id,
                required_capability=required_capability,
            ),
            preflight_snapshot,
            utc_now(),
        )
        if not preflight_decision.accepted and preflight_decision.status_code == 404:
            detail = {
                "error": "no compatible deployment",
                "model_name": model_name,
                "task": task,
                "required_capability": required_capability,
                "last_decision_reason": preflight_decision.reason,
                **self._build_candidate_status(
                    model_name=model_name,
                    task=task,
                    required_capability=required_capability,
                    snapshot=preflight_snapshot,
                ),
            }
            raise RuntimeOperationError("no compatible deployment", detail=detail, status_code=404)
        queued_request = ActiveRequest(
            id=request_id,
            user_name=auth.user_name,
            api_key_id=auth.id,
            deployment_id="",
            backend_runtime_id="",
            gpu_group_id="",
            request_class="interactive",
            model_name=model_name,
            status="queued",
            priority=auth.priority,
            created_at=utc_now_iso(),
        )
        self.state_store.mutate(lambda snapshot: self._enqueue_request(snapshot, queued_request))
        interactive_timeout_seconds = (
            self.config.global_config.effective_interactive_request_timeout_seconds()
        )
        deadline = asyncio.get_event_loop().time() + interactive_timeout_seconds
        last_decision_reason = "no scheduling decision made yet"
        last_detail: dict[str, Any] = {}
        while asyncio.get_event_loop().time() < deadline:
            snapshot = self.snapshot()
            if not any(item.id == request_id for item in snapshot.queued_requests):
                detail = {
                    "error": "request cancelled",
                    "request_id": request_id,
                    "model_name": model_name,
                    "task": task,
                }
                raise RuntimeOperationError("request cancelled", detail=detail, status_code=409)
            decision = self.scheduler.schedule(
                SchedulingRequest(
                    model_name=model_name,
                    task=task,
                    priority=auth.priority,
                    request_class="interactive",
                    request_id=request_id,
                    required_capability=required_capability,
                ),
                snapshot,
                utc_now(),
            )
            last_decision_reason = decision.reason
            last_detail = self._build_candidate_status(
                model_name=model_name,
                task=task,
                required_capability=required_capability,
                snapshot=snapshot,
            )
            if not decision.accepted and decision.status_code == 404:
                self.state_store.mutate(lambda state: self._drop_request(state, request_id))
                detail = {
                    "error": "no compatible deployment",
                    "model_name": model_name,
                    "task": task,
                    "required_capability": required_capability,
                    "last_decision_reason": decision.reason,
                    **last_detail,
                }
                raise RuntimeOperationError(
                    "no compatible deployment",
                    detail=detail,
                    status_code=404,
                )
            if decision.accepted and decision.deployment_id is not None:
                if any(
                    snapshot.deployments.get(deployment_id)
                    and snapshot.deployments[deployment_id].active_request_ids
                    for deployment_id in (decision.should_evict or [])
                ):
                    await asyncio.sleep(self.config.global_config.queue_poll_interval_seconds)
                    continue
                deployment = self.config.deployments[decision.deployment_id]
                try:
                    async for chunk in self._execute_stream_request(
                        deployment=deployment,
                        request_id=request_id,
                        payload=payload,
                        task=task,
                        auth=auth,
                        target_gpu_group_id=(
                            decision.gpu_group_id or deployment.preferred_gpu_group_id()
                        ),
                        should_evict=decision.should_evict or [],
                    ):
                        yield chunk
                    self.state_store.mutate(lambda state: self._dequeue_request(state, request_id))
                    return
                except RuntimeOperationError as exc:
                    diagnostic = exc.detail
                    self.state_store.mutate(
                        lambda state: self._mark_request_diagnostic(
                            state,
                            request_id,
                            diagnostic,
                        )
                    )
                    self.state_store.mutate(lambda state: self._drop_request(state, request_id))
                    raise
            await asyncio.sleep(self.config.global_config.queue_poll_interval_seconds)
            self.refresh_gpu_observations()
            self.enforce_keep_free()
        self.state_store.mutate(lambda state: self._drop_request(state, request_id))
        detail = {
            "error": "request timed out waiting for a deployment",
            "model_name": model_name,
            "task": task,
            "timeout_seconds": interactive_timeout_seconds,
            "last_decision_reason": last_decision_reason,
            **last_detail,
        }
        self.event_logger.emit(
            "routing.timeout",
            "request timed out waiting for a deployment",
            request_id=request_id,
            **detail,
        )
        raise RuntimeOperationError(
            "request timed out waiting for a deployment",
            detail=detail,
            status_code=409,
        )

    async def _execute_request(
        self,
        *,
        deployment: DeploymentConfig,
        request_id: str,
        payload: dict[str, Any],
        task: str,
        auth: AuthResult,
        target_gpu_group_id: str,
        should_evict: list[str],
        uploaded_file: UploadedFilePayload | None,
    ) -> Any:
        snapshot = self.state_store.mutate(self._reconcile_loaded_processes)
        current = snapshot.deployments.get(deployment.id)
        current_gpu_group_id = (
            (current.selected_gpu_group_id or current.gpu_group_id)
            if current is not None
            else None
        )
        should_start = (
            current is None or not current.loaded or current_gpu_group_id != target_gpu_group_id
        )
        planned_evictions = list(should_evict)
        if (
            current is not None
            and current.loaded
            and current_gpu_group_id is not None
            and current_gpu_group_id != target_gpu_group_id
        ):
            if current.active_request_ids:
                raise RuntimeOperationError(
                    "deployment is busy and cannot switch gpu group",
                    detail={
                        "error": "deployment is busy and cannot switch gpu group",
                        "deployment_id": deployment.id,
                        "current_gpu_group_id": current_gpu_group_id,
                        "target_gpu_group_id": target_gpu_group_id,
                        "active_request_ids": current.active_request_ids,
                    },
                    status_code=409,
                )
            planned_evictions.append(deployment.id)
        if should_start:
            planned_evictions = list(dict.fromkeys(planned_evictions))
            self._prepare_group_for_load(planned_evictions, reason=f"routing to {deployment.id}")
            await self._start_and_mark_ready(
                deployment,
                selected_gpu_group_id=target_gpu_group_id,
                reason=f"routing request {request_id}",
            )
        self.state_store.mutate(
            lambda state: self._mark_request_running(
                state,
                request_id,
                deployment,
                auth,
                target_gpu_group_id,
            )
        )
        observed_pid = self.snapshot().deployments[deployment.id].process_id
        recovery_attempts = 0
        while True:
            adapter = self.backends.adapter_for(
                deployment.backend_runtime_id,
                gpu_group_id=target_gpu_group_id,
            )
            try:
                result = await self._invoke_backend_request(
                    adapter,
                    task=task,
                    payload=payload,
                    uploaded_file=uploaded_file,
                )
                self.state_store.mutate(
                    lambda state: self._mark_request_finished(state, request_id, deployment.id)
                )
                return result
            except RuntimeOperationError as exc:
                runtime_error = str(exc)
                self.state_store.mutate(
                    lambda state, runtime_error=runtime_error: self._mark_request_failed(
                        state,
                        request_id,
                        deployment.id,
                        runtime_error,
                    )
                )
                raise
            except (asyncio.CancelledError, GeneratorExit):
                raise
            except Exception as exc:
                if recovery_attempts == 0 and is_backend_connection_error(exc):
                    recovery_attempts = 1
                    connection_error = str(exc)
                    try:
                        await self._recover_backend_for_request(
                            deployment,
                            selected_gpu_group_id=target_gpu_group_id,
                            observed_pid=observed_pid,
                            request_id=request_id,
                            error=connection_error,
                        )
                    except RuntimeOperationError:
                        self.state_store.mutate(
                            lambda state, connection_error=connection_error: (
                                self._mark_request_failed(
                                    state,
                                    request_id,
                                    deployment.id,
                                    connection_error,
                                )
                            )
                        )
                        raise
                    observed_pid = self.snapshot().deployments[deployment.id].process_id
                    continue
                error = str(exc)
                self.state_store.mutate(
                    lambda state, error=error: self._mark_request_failed(
                        state,
                        request_id,
                        deployment.id,
                        error,
                    )
                )
                raise RuntimeOperationError(
                    "backend request failed",
                    detail={
                        "error": "backend request failed",
                        "deployment_id": deployment.id,
                        "backend_runtime_id": deployment.backend_runtime_id,
                        "gpu_group_id": target_gpu_group_id,
                        "detail": error,
                        "recovery_attempts": recovery_attempts,
                    },
                    status_code=409,
                ) from exc

    async def _invoke_backend_request(
        self,
        adapter: Any,
        *,
        task: str,
        payload: dict[str, Any],
        uploaded_file: UploadedFilePayload | None,
    ) -> Any:
        if task == "chat":
            return await adapter.invoke_chat(payload)
        if task == "response":
            return await adapter.invoke_response(payload)
        if task == "completion":
            return await adapter.invoke_completion(payload)
        if task == "embedding":
            return await adapter.invoke_embeddings(payload)
        if task == "audio_speech":
            return await adapter.invoke_audio_speech(payload)
        if task == "audio_transcription":
            if uploaded_file is None:
                raise RuntimeOperationError(
                    "missing audio file for transcription",
                    detail={"error": "missing audio file for transcription"},
                    status_code=422,
                )
            return await adapter.invoke_audio_transcription(payload, uploaded_file)
        if task == "audio_translation":
            if uploaded_file is None:
                raise RuntimeOperationError(
                    "missing audio file for translation",
                    detail={"error": "missing audio file for translation"},
                    status_code=422,
                )
            return await adapter.invoke_audio_translation(payload, uploaded_file)
        return await adapter.invoke_multimodal_operation(
            task,
            payload=payload,
            uploaded_file=uploaded_file,
        )

    async def _execute_stream_request(
        self,
        *,
        deployment: DeploymentConfig,
        request_id: str,
        payload: dict[str, Any],
        task: str,
        auth: AuthResult,
        target_gpu_group_id: str,
        should_evict: list[str],
    ) -> AsyncIterator[bytes]:
        if task not in {"response", "completion"}:
            raise RuntimeOperationError(
                "streaming is only implemented for responses and completions",
                detail={"error": "unsupported streaming task", "task": task},
                status_code=409,
            )
        snapshot = self.state_store.mutate(self._reconcile_loaded_processes)
        current = snapshot.deployments.get(deployment.id)
        current_gpu_group_id = (
            (current.selected_gpu_group_id or current.gpu_group_id)
            if current is not None
            else None
        )
        should_start = (
            current is None or not current.loaded or current_gpu_group_id != target_gpu_group_id
        )
        planned_evictions = list(should_evict)
        if (
            current is not None
            and current.loaded
            and current_gpu_group_id is not None
            and current_gpu_group_id != target_gpu_group_id
        ):
            if current.active_request_ids:
                raise RuntimeOperationError(
                    "deployment is busy and cannot switch gpu group",
                    detail={
                        "error": "deployment is busy and cannot switch gpu group",
                        "deployment_id": deployment.id,
                        "current_gpu_group_id": current_gpu_group_id,
                        "target_gpu_group_id": target_gpu_group_id,
                        "active_request_ids": current.active_request_ids,
                    },
                    status_code=409,
                )
            planned_evictions.append(deployment.id)
        if should_start:
            planned_evictions = list(dict.fromkeys(planned_evictions))
            self._prepare_group_for_load(planned_evictions, reason=f"routing to {deployment.id}")
            await self._start_and_mark_ready(
                deployment,
                selected_gpu_group_id=target_gpu_group_id,
                reason=f"routing request {request_id}",
            )
        self.state_store.mutate(
            lambda state: self._mark_request_running(
                state,
                request_id,
                deployment,
                auth,
                target_gpu_group_id,
            )
        )
        observed_pid = self.snapshot().deployments[deployment.id].process_id
        recovery_attempts = 0
        sent_any_chunk = False
        while True:
            adapter = self.backends.adapter_for(
                deployment.backend_runtime_id,
                gpu_group_id=target_gpu_group_id,
            )
            try:
                stream = (
                    adapter.stream_response(payload)
                    if task == "response"
                    else adapter.stream_completion(payload)
                )
                async for chunk in stream:
                    sent_any_chunk = True
                    yield chunk
                self.state_store.mutate(
                    lambda state: self._mark_request_finished(state, request_id, deployment.id)
                )
                return
            except RuntimeOperationError as exc:
                runtime_error = str(exc)
                self.state_store.mutate(
                    lambda state, runtime_error=runtime_error: self._mark_request_failed(
                        state,
                        request_id,
                        deployment.id,
                        runtime_error,
                    )
                )
                raise
            except (asyncio.CancelledError, GeneratorExit):
                self.state_store.mutate(
                    lambda state: self._mark_request_failed(
                        state,
                        request_id,
                        deployment.id,
                        "stream cancelled",
                    )
                )
                raise
            except Exception as exc:
                if (
                    recovery_attempts == 0
                    and not sent_any_chunk
                    and is_backend_connection_error(exc)
                ):
                    recovery_attempts = 1
                    connection_error = str(exc)
                    try:
                        await self._recover_backend_for_request(
                            deployment,
                            selected_gpu_group_id=target_gpu_group_id,
                            observed_pid=observed_pid,
                            request_id=request_id,
                            error=connection_error,
                        )
                    except RuntimeOperationError:
                        self.state_store.mutate(
                            lambda state, connection_error=connection_error: (
                                self._mark_request_failed(
                                    state,
                                    request_id,
                                    deployment.id,
                                    connection_error,
                                )
                            )
                        )
                        raise
                    observed_pid = self.snapshot().deployments[deployment.id].process_id
                    continue
                error = str(exc)
                self.state_store.mutate(
                    lambda state, error=error: self._mark_request_failed(
                        state,
                        request_id,
                        deployment.id,
                        error,
                    )
                )
                raise RuntimeOperationError(
                    "backend streaming request failed",
                    detail={
                        "error": "backend streaming request failed",
                        "deployment_id": deployment.id,
                        "backend_runtime_id": deployment.backend_runtime_id,
                        "gpu_group_id": target_gpu_group_id,
                        "detail": error,
                        "recovery_attempts": recovery_attempts,
                        "stream_started": sent_any_chunk,
                    },
                    status_code=409,
                ) from exc

    def _prepare_group_for_load(self, deployment_ids: list[str], *, reason: str) -> None:
        snapshot = self.snapshot()
        for deployment_id in deployment_ids:
            state = snapshot.deployments.get(deployment_id)
            if state is not None and state.loaded and not state.active_request_ids:
                self.event_logger.emit(
                    "deployment.evict",
                    "evicting deployment before loading target deployment",
                    deployment_id=deployment_id,
                    gpu_group_id=state.selected_gpu_group_id or state.gpu_group_id,
                    reason=reason,
                )
                self._ensure_supervised_process(deployment_id, state)
                self.backends.stop(
                    deployment_id,
                    gpu_group_id=state.selected_gpu_group_id or state.gpu_group_id,
                    force=False,
                )
                self.state_store.mutate(
                    lambda current: self._mark_unloaded(current, deployment_id, reason=reason)
                )

    def _ensure_supervised_process(self, deployment_id: str, runtime_state: DeploymentRuntimeState) -> None:
        if deployment_id in self.backends.supervisor.processes:
            return
        if runtime_state.process_id is None:
            return
        if runtime_state.backend_runtime_id not in self.config.backends:
            return
        gpu_group_id = runtime_state.selected_gpu_group_id or runtime_state.gpu_group_id
        backend = self.backends.resolve_backend(runtime_state.backend_runtime_id, gpu_group_id=gpu_group_id)
        self.backends.supervisor.adopt(
            deployment_id=deployment_id,
            pid=runtime_state.process_id,
            command=backend.launch_command,
            log_path=self.state_root / "logs" / f"{deployment_id}.log",
            started_at=runtime_state.loaded_at or utc_now_iso(),
        )

    async def _stop_unhealthy_backend(
        self,
        deployment: DeploymentConfig,
        runtime_state: DeploymentRuntimeState,
        *,
        gpu_group_id: str,
    ) -> None:
        if runtime_state.process_id is None:
            return
        self._ensure_supervised_process(deployment.id, runtime_state)
        try:
            await asyncio.to_thread(
                self.backends.stop,
                deployment.id,
                gpu_group_id=gpu_group_id,
                force=False,
            )
        except Exception:
            await asyncio.to_thread(
                self.backends.stop,
                deployment.id,
                gpu_group_id=gpu_group_id,
                force=True,
            )

    async def _invalidate_unhealthy_backend(
        self,
        deployment: DeploymentConfig,
        *,
        selected_gpu_group_id: str,
        observed_pid: int | None,
        reason: str,
        source: str,
    ) -> None:
        async with self._deployment_lock(deployment.id):
            current = self.snapshot().deployments.get(deployment.id)
            if current is None:
                return
            current_pid = current.process_id
            if (
                observed_pid is not None
                and current.loaded
                and current_pid is not None
                and current_pid != observed_pid
            ):
                self.event_logger.emit(
                    "backend.recovery",
                    "ignored a stale backend failure signal",
                    deployment_id=deployment.id,
                    backend_runtime_id=deployment.backend_runtime_id,
                    gpu_group_id=selected_gpu_group_id,
                    old_pid=observed_pid,
                    reason=reason,
                    recovery_result="newer_backend_already_ready",
                    source=source,
                    new_pid=current_pid,
                )
                return
            if not current.loaded and current.process_id is None:
                return
            old_pid = current.process_id or observed_pid
            try:
                await self._stop_unhealthy_backend(
                    deployment,
                    current,
                    gpu_group_id=selected_gpu_group_id,
                )
            except Exception as exc:
                cleanup_error = str(exc)
                self.state_store.mutate(
                    lambda state, cleanup_error=cleanup_error: self._mark_unavailable(
                        state,
                        deployment.id,
                        reason=reason,
                        error=f"failed to stop unhealthy backend: {cleanup_error}",
                    )
                )
                self.event_logger.emit(
                    "backend.recovery",
                    "failed to clean up unhealthy backend",
                    deployment_id=deployment.id,
                    backend_runtime_id=deployment.backend_runtime_id,
                    gpu_group_id=selected_gpu_group_id,
                    old_pid=old_pid,
                    reason=reason,
                    recovery_result="cleanup_failed",
                    source=source,
                    error=str(exc),
                )
                raise
            self.state_store.mutate(
                lambda state: self._mark_unavailable(
                    state,
                    deployment.id,
                    reason=reason,
                    error=reason,
                    preserve_active_requests=source == "request",
                )
            )
            self.event_logger.emit(
                "backend.recovery",
                "cleaned up unhealthy backend and marked it unavailable",
                deployment_id=deployment.id,
                backend_runtime_id=deployment.backend_runtime_id,
                gpu_group_id=selected_gpu_group_id,
                old_pid=old_pid,
                reason=reason,
                recovery_result="marked_unavailable",
                source=source,
            )

    async def _recover_backend_for_request(
        self,
        deployment: DeploymentConfig,
        *,
        selected_gpu_group_id: str,
        observed_pid: int | None,
        request_id: str,
        error: str,
    ) -> None:
        reason = f"request {request_id} connection failure: {error}"
        async with self._deployment_lock(deployment.id):
            current = self.snapshot().deployments.get(deployment.id)
            current_pid = current.process_id if current is not None else None
            if (
                observed_pid is not None
                and current is not None
                and current.loaded
                and current_pid is not None
                and current_pid != observed_pid
            ):
                self.event_logger.emit(
                    "backend.recovery",
                    "reused backend recovered by another request",
                    deployment_id=deployment.id,
                    backend_runtime_id=deployment.backend_runtime_id,
                    gpu_group_id=selected_gpu_group_id,
                    old_pid=observed_pid,
                    reason=reason,
                    recovery_result="newer_backend_already_ready",
                    source="request",
                    request_id=request_id,
                    new_pid=current_pid,
                )
                return
            old_pid = current_pid or observed_pid
            self.event_logger.emit(
                "backend.recovery_started",
                "recovering backend after request connection failure",
                deployment_id=deployment.id,
                backend_runtime_id=deployment.backend_runtime_id,
                gpu_group_id=selected_gpu_group_id,
                old_pid=old_pid,
                reason=reason,
                recovery_result="started",
                source="request",
                request_id=request_id,
            )
            try:
                if current is not None and (current.loaded or current.process_id is not None):
                    await self._stop_unhealthy_backend(
                        deployment,
                        current,
                        gpu_group_id=selected_gpu_group_id,
                    )
                self.state_store.mutate(
                    lambda state: self._mark_unavailable(
                        state,
                        deployment.id,
                        reason=reason,
                        error=error,
                        preserve_active_requests=True,
                    )
                )
                await self._start_and_mark_ready_locked(
                    deployment,
                    selected_gpu_group_id=selected_gpu_group_id,
                    reason=f"automatic recovery for request {request_id}",
                )
            except Exception as exc:
                self.event_logger.emit(
                    "backend.recovery_failed",
                    "backend recovery failed",
                    deployment_id=deployment.id,
                    backend_runtime_id=deployment.backend_runtime_id,
                    gpu_group_id=selected_gpu_group_id,
                    old_pid=old_pid,
                    reason=reason,
                    recovery_result="failed",
                    source="request",
                    request_id=request_id,
                    error=str(exc),
                )
                detail = exc.detail if isinstance(exc, RuntimeOperationError) else {"error": str(exc)}
                raise RuntimeOperationError(
                    "backend recovery failed",
                    detail={
                        "error": "backend recovery failed",
                        "deployment_id": deployment.id,
                        "backend_runtime_id": deployment.backend_runtime_id,
                        "gpu_group_id": selected_gpu_group_id,
                        "old_pid": old_pid,
                        "reason": reason,
                        "recovery_attempts": 1,
                        "recovery": detail,
                    },
                    status_code=409,
                ) from exc
            new_pid = self.snapshot().deployments[deployment.id].process_id
            self.event_logger.emit(
                "backend.recovery_succeeded",
                "backend recovered and passed readiness",
                deployment_id=deployment.id,
                backend_runtime_id=deployment.backend_runtime_id,
                gpu_group_id=selected_gpu_group_id,
                old_pid=old_pid,
                new_pid=new_pid,
                reason=reason,
                recovery_result="ready",
                source="request",
                request_id=request_id,
            )

    def _reconcile_loaded_processes(self, snapshot: RuntimeStateSnapshot) -> RuntimeStateSnapshot:
        snapshot = self._seed_snapshot(snapshot)
        for deployment_id, runtime_state in snapshot.deployments.items():
            if not runtime_state.loaded:
                continue
            if runtime_state.process_id is None:
                reason = "state reconciliation: missing process_id for loaded deployment"
                self._mark_unavailable(
                    snapshot,
                    deployment_id,
                    reason=reason,
                    error="tracked backend process is missing",
                )
                self.event_logger.emit(
                    "backend.recovery",
                    "marked deployment unavailable during process reconciliation",
                    deployment_id=deployment_id,
                    backend_runtime_id=runtime_state.backend_runtime_id,
                    gpu_group_id=runtime_state.selected_gpu_group_id or runtime_state.gpu_group_id,
                    old_pid=None,
                    reason=reason,
                    recovery_result="marked_unavailable",
                )
                continue
            try:
                os.kill(runtime_state.process_id, 0)
            except OSError:
                old_pid = runtime_state.process_id
                reason = "state reconciliation: tracked process is not running"
                self.backends.supervisor.processes.pop(deployment_id, None)
                self._mark_unavailable(
                    snapshot,
                    deployment_id,
                    reason=reason,
                    error="tracked backend process is not running",
                )
                self.event_logger.emit(
                    "backend.recovery",
                    "marked deployment unavailable during process reconciliation",
                    deployment_id=deployment_id,
                    backend_runtime_id=runtime_state.backend_runtime_id,
                    gpu_group_id=runtime_state.selected_gpu_group_id or runtime_state.gpu_group_id,
                    old_pid=old_pid,
                    reason=reason,
                    recovery_result="marked_unavailable",
                )
                continue
            self._ensure_supervised_process(deployment_id, runtime_state)
        return snapshot

    def _enqueue_request(self, snapshot: RuntimeStateSnapshot, request: ActiveRequest) -> RuntimeStateSnapshot:
        snapshot = self._seed_snapshot(snapshot)
        snapshot.queued_requests.append(request)
        return snapshot

    def _dequeue_request(self, snapshot: RuntimeStateSnapshot, request_id: str) -> RuntimeStateSnapshot:
        snapshot.queued_requests = [item for item in snapshot.queued_requests if item.id != request_id]
        return snapshot

    def _drop_request(self, snapshot: RuntimeStateSnapshot, request_id: str) -> RuntimeStateSnapshot:
        snapshot.queued_requests = [item for item in snapshot.queued_requests if item.id != request_id]
        snapshot.active_requests.pop(request_id, None)
        return snapshot

    def _mark_request_diagnostic(
        self,
        snapshot: RuntimeStateSnapshot,
        request_id: str,
        detail: dict[str, Any],
    ) -> RuntimeStateSnapshot:
        for item in snapshot.queued_requests:
            if item.id == request_id:
                item.detail = detail
                break
        return snapshot

    async def _start_and_mark_ready(
        self,
        deployment: DeploymentConfig,
        *,
        selected_gpu_group_id: str,
        reason: str,
    ) -> None:
        async with self._deployment_lock(deployment.id):
            await self._start_and_mark_ready_locked(
                deployment,
                selected_gpu_group_id=selected_gpu_group_id,
                reason=reason,
            )

    async def _start_and_mark_ready_locked(
        self,
        deployment: DeploymentConfig,
        *,
        selected_gpu_group_id: str,
        reason: str,
    ) -> None:
        current = self.snapshot().deployments.get(deployment.id)
        if (
            current is not None
            and current.loaded
            and (current.selected_gpu_group_id or current.gpu_group_id)
            == selected_gpu_group_id
        ):
            return
        self.state_store.mutate(
            lambda state: self._mark_starting(
                state,
                deployment,
                selected_gpu_group_id=selected_gpu_group_id,
                reason=reason,
            )
        )
        try:
            readiness = await self.backends.ensure_started_and_ready(
                deployment,
                gpu_group_id=selected_gpu_group_id,
            )
        except BackendOperationError as exc:
            readiness_detail = exc.detail
            self.state_store.mutate(
                lambda state: self._mark_start_failed(
                    state,
                    deployment,
                    selected_gpu_group_id=selected_gpu_group_id,
                    detail=readiness_detail,
                    reason=reason,
                )
            )
            raise RuntimeOperationError(
                "backend failed to start cleanly",
                detail={
                    "error": "backend failed to start cleanly",
                    "deployment_id": deployment.id,
                    "gpu_group_id": selected_gpu_group_id,
                    "backend_runtime_id": deployment.backend_runtime_id,
                    "reason": reason,
                    "readiness": readiness_detail,
                },
                status_code=409,
            ) from exc
        self.state_store.mutate(
            lambda state: self._mark_loaded(
                state,
                deployment,
                selected_gpu_group_id=selected_gpu_group_id,
                pid=readiness["pid"],
                reason=reason,
                readiness_detail=readiness,
            )
        )

    def _mark_loaded(
        self,
        snapshot: RuntimeStateSnapshot,
        deployment: DeploymentConfig,
        *,
        selected_gpu_group_id: str,
        pid: int,
        reason: str,
        readiness_detail: dict[str, Any],
    ) -> RuntimeStateSnapshot:
        snapshot = self._seed_snapshot(snapshot)
        runtime = snapshot.deployments[deployment.id]
        runtime.loaded = True
        runtime.gpu_group_id = selected_gpu_group_id
        runtime.selected_gpu_group_id = selected_gpu_group_id
        runtime.state = "ready"
        runtime.desired_state = "loaded"
        runtime.loaded_at = utc_now_iso()
        runtime.readiness_passed_at = readiness_detail.get("ready_at", utc_now_iso())
        runtime.process_id = pid
        runtime.resident_memory_fraction = deployment.memory_fraction_for_group(selected_gpu_group_id)
        runtime.current_model_name = deployment.api_model_name
        runtime.last_error = None
        runtime.last_transition_reason = reason
        runtime.last_readiness_detail = readiness_detail
        return snapshot

    def _mark_starting(
        self,
        snapshot: RuntimeStateSnapshot,
        deployment: DeploymentConfig,
        *,
        selected_gpu_group_id: str,
        reason: str,
    ) -> RuntimeStateSnapshot:
        snapshot = self._seed_snapshot(snapshot)
        runtime = snapshot.deployments[deployment.id]
        runtime.gpu_group_id = selected_gpu_group_id
        runtime.selected_gpu_group_id = selected_gpu_group_id
        runtime.state = "starting"
        runtime.desired_state = "loaded"
        runtime.startup_started_at = utc_now_iso()
        runtime.last_transition_reason = reason
        runtime.current_model_name = deployment.api_model_name
        runtime.last_error = None
        return snapshot

    def _mark_start_failed(
        self,
        snapshot: RuntimeStateSnapshot,
        deployment: DeploymentConfig,
        *,
        selected_gpu_group_id: str,
        detail: dict[str, Any],
        reason: str,
    ) -> RuntimeStateSnapshot:
        snapshot = self._seed_snapshot(snapshot)
        runtime = snapshot.deployments[deployment.id]
        runtime.loaded = False
        runtime.gpu_group_id = selected_gpu_group_id
        runtime.selected_gpu_group_id = None
        runtime.state = "failed"
        runtime.desired_state = "unloaded"
        runtime.process_id = None
        runtime.resident_memory_fraction = 0.0
        runtime.last_error = detail.get("error")
        runtime.last_transition_reason = reason
        runtime.last_readiness_detail = detail
        return snapshot

    def _mark_unavailable(
        self,
        snapshot: RuntimeStateSnapshot,
        deployment_id: str,
        *,
        reason: str,
        error: str,
        preserve_active_requests: bool = False,
    ) -> RuntimeStateSnapshot:
        runtime = snapshot.deployments[deployment_id]
        runtime.loaded = False
        runtime.state = "failed"
        runtime.desired_state = "unloaded"
        runtime.selected_gpu_group_id = None
        runtime.process_id = None
        runtime.resident_memory_fraction = 0.0
        if not preserve_active_requests:
            runtime.active_request_ids = []
        runtime.last_error = error
        runtime.last_transition_reason = reason
        return snapshot

    def _mark_unloaded(
        self,
        snapshot: RuntimeStateSnapshot,
        deployment_id: str,
        *,
        reason: str,
    ) -> RuntimeStateSnapshot:
        runtime = snapshot.deployments[deployment_id]
        runtime.loaded = False
        runtime.state = "unloaded"
        runtime.desired_state = "unloaded"
        runtime.selected_gpu_group_id = None
        runtime.process_id = None
        runtime.resident_memory_fraction = 0.0
        runtime.active_request_ids = []
        runtime.last_transition_reason = reason
        return snapshot

    def _mark_request_running(
        self,
        snapshot: RuntimeStateSnapshot,
        request_id: str,
        deployment: DeploymentConfig,
        auth: AuthResult,
        selected_gpu_group_id: str,
    ) -> RuntimeStateSnapshot:
        snapshot = self._seed_snapshot(snapshot)
        queued = next((item for item in snapshot.queued_requests if item.id == request_id), None)
        if queued is None:
            raise RuntimeOperationError(
                "request cancelled",
                detail={"error": "request cancelled", "request_id": request_id},
                status_code=409,
            )
        queued.deployment_id = deployment.id
        queued.backend_runtime_id = deployment.backend_runtime_id
        queued.gpu_group_id = selected_gpu_group_id
        queued.status = "running"
        queued.started_at = utc_now_iso()
        snapshot.active_requests[request_id] = queued
        runtime = snapshot.deployments[deployment.id]
        runtime.gpu_group_id = selected_gpu_group_id
        runtime.selected_gpu_group_id = selected_gpu_group_id
        if request_id not in runtime.active_request_ids:
            runtime.active_request_ids.append(request_id)
        runtime.last_used_at = utc_now_iso()
        runtime.state = "ready"
        self.event_logger.audit(
            "request.started",
            auth.user_name,
            request_id=request_id,
            deployment_id=deployment.id,
            gpu_group_id=selected_gpu_group_id,
        )
        return snapshot

    def _mark_request_finished(
        self,
        snapshot: RuntimeStateSnapshot,
        request_id: str,
        deployment_id: str,
    ) -> RuntimeStateSnapshot:
        request = snapshot.active_requests.pop(request_id, None)
        if request is not None:
            request.status = "completed"
            request.finished_at = utc_now_iso()
        runtime = snapshot.deployments[deployment_id]
        runtime.active_request_ids = [item for item in runtime.active_request_ids if item != request_id]
        runtime.last_used_at = utc_now_iso()
        if runtime.loaded:
            runtime.state = "ready"
        self.event_logger.emit(
            "request.completed",
            "request finished",
            request_id=request_id,
            deployment_id=deployment_id,
        )
        return snapshot

    def _mark_request_failed(
        self,
        snapshot: RuntimeStateSnapshot,
        request_id: str,
        deployment_id: str,
        error: str,
    ) -> RuntimeStateSnapshot:
        request = snapshot.active_requests.pop(request_id, None)
        if request is not None:
            request.status = "failed"
            request.finished_at = utc_now_iso()
            request.error = error
        runtime = snapshot.deployments[deployment_id]
        runtime.active_request_ids = [item for item in runtime.active_request_ids if item != request_id]
        runtime.last_used_at = utc_now_iso()
        if runtime.loaded:
            runtime.state = "ready"
        runtime.last_error = error
        self.event_logger.emit(
            "request.failed",
            "request failed",
            request_id=request_id,
            deployment_id=deployment_id,
            error=error,
        )
        return snapshot

    async def submit_batch(self, request: BatchCreateRequest, auth: AuthResult) -> BatchJobState:
        job = BatchJobState(
            id=f"batch_{uuid.uuid4().hex}",
            api_key_id=auth.id,
            user_name=auth.user_name,
            model_name=request.model,
            status="queued",
            created_at=utc_now_iso(),
            updated_at=utc_now_iso(),
            total_items=len(request.requests),
            metadata={"requests": request.requests, **request.metadata},
        )
        self.state_store.mutate(lambda snapshot: self._add_batch(snapshot, job))
        self.event_logger.emit("batch.queued", "batch job queued", batch_id=job.id, model_name=request.model)
        return job

    def _add_batch(self, snapshot: RuntimeStateSnapshot, job: BatchJobState) -> RuntimeStateSnapshot:
        snapshot.batch_jobs[job.id] = job
        return snapshot

    async def process_one_batch(self) -> None:
        self.refresh_gpu_observations()
        self.enforce_keep_free()
        snapshot = self.snapshot()
        queued_jobs = [job for job in snapshot.batch_jobs.values() if job.status == "queued"]
        if not queued_jobs:
            return
        job = sorted(queued_jobs, key=lambda item: item.created_at)[0]
        decision = self.scheduler.schedule(
            SchedulingRequest(
                model_name=job.model_name,
                task="chat",
                priority=100,
                request_class="batch",
                request_id=job.id,
                required_capability="text",
            ),
            snapshot,
            utc_now(),
        )
        if not decision.accepted or decision.deployment_id is None:
            return
        deployment = self.config.deployments[decision.deployment_id]
        selected_gpu_group_id = decision.gpu_group_id or deployment.preferred_gpu_group_id()
        if decision.should_load:
            self._prepare_group_for_load(
                decision.should_evict or [],
                reason=f"batch scheduling for {job.id}",
            )
            await self._start_and_mark_ready(
                deployment,
                selected_gpu_group_id=selected_gpu_group_id,
                reason=f"batch scheduling for {job.id}",
            )
        adapter = self.backends.adapter_for(
            deployment.backend_runtime_id,
            gpu_group_id=selected_gpu_group_id,
        )
        self.state_store.mutate(lambda state: self._mark_batch_running(state, job.id, deployment.id))
        requests = job.metadata.get("requests", [])
        completed = 0
        failed = 0
        for item in requests:
            try:
                await adapter.invoke_chat(item)
                completed += 1
            except Exception:
                failed += 1
            self.state_store.mutate(
                lambda state: self._update_batch_progress(state, job.id, completed, failed)
            )
        self.state_store.mutate(lambda state: self._finish_batch(state, job.id, completed, failed))

    def _mark_batch_running(
        self,
        snapshot: RuntimeStateSnapshot,
        batch_id: str,
        deployment_id: str,
    ) -> RuntimeStateSnapshot:
        job = snapshot.batch_jobs[batch_id]
        job.status = "running"
        job.updated_at = utc_now_iso()
        job.deployment_id = deployment_id
        return snapshot

    def _update_batch_progress(
        self,
        snapshot: RuntimeStateSnapshot,
        batch_id: str,
        completed: int,
        failed: int,
    ) -> RuntimeStateSnapshot:
        job = snapshot.batch_jobs[batch_id]
        job.completed_items = completed
        job.failed_items = failed
        job.updated_at = utc_now_iso()
        return snapshot

    def _finish_batch(
        self,
        snapshot: RuntimeStateSnapshot,
        batch_id: str,
        completed: int,
        failed: int,
    ) -> RuntimeStateSnapshot:
        job = snapshot.batch_jobs[batch_id]
        job.status = "completed" if failed == 0 else "failed"
        job.completed_items = completed
        job.failed_items = failed
        job.updated_at = utc_now_iso()
        return snapshot

    async def drain_group(self, gpu_group_id: str, timeout_seconds: int) -> DrainState:
        started = utc_now_iso()
        drain = DrainState(
            gpu_group_id=gpu_group_id,
            status="pending",
            started_at=started,
            timeout_seconds=timeout_seconds,
        )
        self.state_store.mutate(lambda snapshot: self._start_drain(snapshot, drain))
        deadline = asyncio.get_event_loop().time() + timeout_seconds
        while asyncio.get_event_loop().time() < deadline:
            snapshot = self.snapshot()
            loaded_in_group = [
                deployment_id
                for deployment_id, state in snapshot.deployments.items()
                if state.gpu_group_id == gpu_group_id and state.loaded
            ]
            active = [
                deployment_id
                for deployment_id in loaded_in_group
                if snapshot.deployments[deployment_id].active_request_ids
            ]
            if not active:
                for deployment_id in loaded_in_group:
                    runtime_state = snapshot.deployments[deployment_id]
                    self.backends.stop(
                        deployment_id,
                        gpu_group_id=runtime_state.selected_gpu_group_id or runtime_state.gpu_group_id,
                        force=True,
                    )
                    self.state_store.mutate(
                        lambda state: self._mark_unloaded(
                            state,
                            deployment_id,
                            reason=f"drain completed for {gpu_group_id}",
                        )
                    )
                completed = drain.model_copy(update={"status": "completed", "completed_at": utc_now_iso()})
                self.state_store.mutate(lambda state: self._finish_drain(state, completed))
                self.event_logger.emit("drain.completed", "gpu group drained", gpu_group_id=gpu_group_id)
                return completed
            await asyncio.sleep(0.25)
        snapshot = self.snapshot()
        loaded_in_group = [
            deployment_id
            for deployment_id, state in snapshot.deployments.items()
            if state.gpu_group_id == gpu_group_id and state.loaded
        ]
        for deployment_id in loaded_in_group:
            runtime_state = snapshot.deployments[deployment_id]
            self.backends.stop(
                deployment_id,
                gpu_group_id=runtime_state.selected_gpu_group_id or runtime_state.gpu_group_id,
                force=True,
            )
            self.state_store.mutate(
                lambda state: self._mark_unloaded(
                    state,
                    deployment_id,
                    reason=f"drain forced for {gpu_group_id}",
                )
            )
        forced = drain.model_copy(update={"status": "forced", "completed_at": utc_now_iso()})
        self.state_store.mutate(lambda state: self._finish_drain(state, forced))
        self.event_logger.emit("drain.forced", "drain timed out and force killed group", gpu_group_id=gpu_group_id)
        return forced

    def _start_drain(self, snapshot: RuntimeStateSnapshot, drain: DrainState) -> RuntimeStateSnapshot:
        snapshot.drains[drain.gpu_group_id] = drain
        return snapshot

    def _finish_drain(self, snapshot: RuntimeStateSnapshot, drain: DrainState) -> RuntimeStateSnapshot:
        snapshot.drains[drain.gpu_group_id] = drain
        return snapshot

    def _set_backend_health(
        self,
        snapshot: RuntimeStateSnapshot,
        health: dict[str, dict[str, Any]],
    ) -> RuntimeStateSnapshot:
        snapshot.backend_health = health
        return snapshot

    def _build_candidate_status(
        self,
        *,
        model_name: str,
        task: str,
        required_capability: Literal["text", "audio", "image", "video"] | None = None,
        snapshot: RuntimeStateSnapshot,
    ) -> dict[str, Any]:
        candidates = [
            deployment
            for deployment in self.config.deployments.values()
            if deployment.enabled
            and deployment.api_model_name == model_name
            and self._deployment_supports_task(deployment, task)
            and (
                required_capability is None
                or required_capability in self._deployment_effective_capabilities(deployment)
            )
        ]
        candidate_details: list[dict[str, Any]] = []
        for deployment in candidates:
            runtime = snapshot.deployments.get(deployment.id)
            model = self.config.models.get(deployment.model_id)
            backend = self.config.backends.get(deployment.backend_runtime_id)
            selected_gpu_group_id = (
                (runtime.selected_gpu_group_id or runtime.gpu_group_id)
                if runtime is not None
                else None
            )
            candidate_details.append(
                {
                    "deployment_id": deployment.id,
                    "gpu_group_id": deployment.preferred_gpu_group_id(),
                    "gpu_group_ids": deployment.eligible_gpu_group_ids(),
                    "selected_gpu_group_id": selected_gpu_group_id,
                    "backend_runtime_id": deployment.backend_runtime_id,
                    "model_capabilities": model.model_capabilities if model is not None else [],
                    "deployment_capabilities": deployment.deployment_capabilities,
                    "backend_modalities": (
                        backend.capabilities.modalities
                        if backend is not None
                        else []
                    ),
                    "effective_capabilities": self._deployment_effective_capabilities(deployment),
                    "loaded": runtime.loaded if runtime is not None else False,
                    "state": runtime.state if runtime is not None else "unloaded",
                    "active_request_ids": runtime.active_request_ids if runtime is not None else [],
                    "last_error": runtime.last_error if runtime is not None else None,
                    "last_transition_reason": runtime.last_transition_reason if runtime is not None else None,
                    "gpu_group_statuses": {
                        gpu_group_id: self._gpu_group_summary(gpu_group_id, snapshot)
                        for gpu_group_id in deployment.eligible_gpu_group_ids()
                        if gpu_group_id in self.config.gpu_groups
                    },
                }
            )
        return {"candidate_deployments": candidate_details}

    def _deployment_effective_capabilities(
        self,
        deployment: DeploymentConfig,
    ) -> list[Literal["text", "audio", "image", "video"]]:
        model = self.config.models.get(deployment.model_id)
        backend = self.config.backends.get(deployment.backend_runtime_id)
        model_capabilities = set(model.model_capabilities if model is not None else [])
        backend_modalities = set(backend.capabilities.modalities if backend is not None else [])
        effective = model_capabilities & backend_modalities
        if deployment.deployment_capabilities:
            effective &= set(deployment.deployment_capabilities)
        return sorted(effective)

    def _gpu_group_summary(
        self,
        gpu_group_id: str,
        snapshot: RuntimeStateSnapshot,
    ) -> dict[str, Any]:
        group = self.config.gpu_groups[gpu_group_id]
        loaded_deployments = [
            deployment_id
            for deployment_id, state in snapshot.deployments.items()
            if state.gpu_group_id == gpu_group_id and state.loaded
        ]
        observations = [snapshot.gpu_observations.get(gpu_id) for gpu_id in group.gpu_ids]
        external_processes = [
            process.model_dump(mode="json")
            for observation in observations
            if observation is not None
            for process in observation.observed_processes
            if not process.managed_by_shardon
        ]
        drain = snapshot.drains.get(gpu_group_id)
        return {
            "gpu_group_id": gpu_group_id,
            "keep_free": group.keep_free,
            "loaded_deployments": loaded_deployments,
            "drain_status": drain.status if drain is not None else None,
            "external_processes": external_processes,
        }

    def _group_runtime_status(self, snapshot: RuntimeStateSnapshot) -> dict[str, Any]:
        return {
            group_id: self._gpu_group_summary(group_id, snapshot)
            for group_id in self.config.gpu_groups
        }

    def _deployment_capability_status(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        for deployment in self.config.deployments.values():
            model = self.config.models.get(deployment.model_id)
            backend = self.config.backends.get(deployment.backend_runtime_id)
            data[deployment.id] = {
                "model_capabilities": model.model_capabilities if model is not None else [],
                "deployment_capabilities": deployment.deployment_capabilities,
                "backend_modalities": backend.capabilities.modalities if backend is not None else [],
                "effective_capabilities": self._deployment_effective_capabilities(deployment),
            }
        return data

    def status(self) -> dict[str, Any]:
        snapshot = self.snapshot()
        return {
            "instance": self.config.global_config.instance_name,
            "deployments": snapshot.deployments,
            "active_requests": snapshot.active_requests,
            "queued_requests": snapshot.queued_requests,
            "batch_jobs": snapshot.batch_jobs,
            "drains": snapshot.drains,
            "gpu_observations": snapshot.gpu_observations,
            "gpu_groups": self._group_runtime_status(snapshot),
            "backend_health": snapshot.backend_health,
            "deployment_capabilities": self._deployment_capability_status(),
        }
