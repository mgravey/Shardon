from __future__ import annotations

import asyncio
import getpass
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from shardon_core.api.schemas import BatchCreateRequest, ChatCompletionRequest, CompletionRequest, EmbeddingRequest
from shardon_core.auth.service import APIKeyService, AdminAuthService, AuthResult
from shardon_core.backends.registry import BackendRegistry
from shardon_core.config.loader import load_repository_config
from shardon_core.config.schemas import DeploymentConfig, RepositoryConfig
from shardon_core.config.writer import delete_yaml, write_yaml
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


class ShardonRuntime:
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
        self.service_user = getpass.getuser()

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
            write_yaml(enabled_path, payload)
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
            snapshot.deployments.setdefault(
                deployment.id,
                DeploymentRuntimeState(
                    deployment_id=deployment.id,
                    gpu_group_id=deployment.gpu_group_id,
                    backend_runtime_id=deployment.backend_runtime_id,
                ),
            )
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
                    self.backends.stop(deployment_id, force=True)
                    state.loaded = False
                    state.process_id = None
                    state.keep_free_killed_at = utc_now_iso()
                    state.active_request_ids = []
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
        models: dict[str, dict[str, Any]] = {}
        for deployment in self.config.deployments.values():
            if not deployment.enabled:
                continue
            model = self.config.models[deployment.model_id]
            models.setdefault(
                deployment.api_model_name,
                {
                    "id": deployment.api_model_name,
                    "object": "model",
                    "owned_by": "shardon",
                    "display_name": deployment.display_name,
                    "source_model_id": model.id,
                    "backend_runtime_id": deployment.backend_runtime_id,
                    "gpu_group_id": deployment.gpu_group_id,
                    "tasks": deployment.tasks,
                },
            )
        return list(models.values())

    async def refresh_backend_health(self) -> RuntimeStateSnapshot:
        results: dict[str, dict[str, Any]] = {}
        for backend_runtime_id in self.config.backends:
            try:
                payload = await self.backends.health(backend_runtime_id)
                results[backend_runtime_id] = {
                    "ok": True,
                    "checked_at": utc_now_iso(),
                    "payload": payload,
                }
            except Exception as exc:
                results[backend_runtime_id] = {
                    "ok": False,
                    "checked_at": utc_now_iso(),
                    "error": str(exc),
                }
                self.event_logger.emit(
                    "backend.health_failed",
                    "backend health check failed",
                    backend_runtime_id=backend_runtime_id,
                    error=str(exc),
                )
        return self.state_store.mutate(lambda snapshot: self._set_backend_health(snapshot, results))

    async def route_chat(
        self,
        request: ChatCompletionRequest,
        auth: AuthResult,
    ) -> dict[str, Any]:
        return await self._route_interactive("chat", request.model, request.model_dump(mode="json"), auth)

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
        )

    async def route_embedding(
        self,
        request: EmbeddingRequest,
        auth: AuthResult,
    ) -> dict[str, Any]:
        return await self._route_interactive("embedding", request.model, request.model_dump(mode="json"), auth)

    async def _route_interactive(
        self,
        task: str,
        model_name: str,
        payload: dict[str, Any],
        auth: AuthResult,
    ) -> dict[str, Any]:
        self.refresh_gpu_observations()
        self.enforce_keep_free()
        request_id = f"req_{uuid.uuid4().hex}"
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
        deadline = asyncio.get_event_loop().time() + 15.0
        while asyncio.get_event_loop().time() < deadline:
            snapshot = self.snapshot()
            decision = self.scheduler.schedule(
                SchedulingRequest(
                    model_name=model_name,
                    task=task,
                    priority=auth.priority,
                    request_class="interactive",
                    request_id=request_id,
                ),
                snapshot,
                utc_now(),
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
                response = await self._execute_request(
                    deployment=deployment,
                    request_id=request_id,
                    payload=payload,
                    task=task,
                    auth=auth,
                    should_evict=decision.should_evict or [],
                )
                self.state_store.mutate(lambda state: self._dequeue_request(state, request_id))
                return response
            await asyncio.sleep(self.config.global_config.queue_poll_interval_seconds)
            self.refresh_gpu_observations()
            self.enforce_keep_free()
        self.state_store.mutate(lambda state: self._drop_request(state, request_id))
        raise RuntimeError("request queued but no deployment became available before timeout")

    async def _execute_request(
        self,
        *,
        deployment: DeploymentConfig,
        request_id: str,
        payload: dict[str, Any],
        task: str,
        auth: AuthResult,
        should_evict: list[str],
    ) -> dict[str, Any]:
        snapshot = self.snapshot()
        current = snapshot.deployments.get(deployment.id)
        if current is None or not current.loaded:
            self._prepare_group_for_load(should_evict)
            pid = self.backends.ensure_started(deployment)
            self.state_store.mutate(
                lambda state: self._mark_loaded(state, deployment, pid)
            )
            await asyncio.sleep(0.5)
        self.state_store.mutate(
            lambda state: self._mark_request_running(state, request_id, deployment, auth)
        )
        adapter = self.backends.adapter_for(deployment.backend_runtime_id)
        try:
            if task == "chat":
                result = await adapter.invoke_chat(payload)
            elif task == "completion":
                result = await adapter.invoke_completion(payload)
            else:
                result = await adapter.invoke_embeddings(payload)
            self.state_store.mutate(lambda state: self._mark_request_finished(state, request_id, deployment.id))
            return result
        except Exception as exc:
            self.state_store.mutate(
                lambda state: self._mark_request_failed(state, request_id, deployment.id, str(exc))
            )
            raise

    def _prepare_group_for_load(self, deployment_ids: list[str]) -> None:
        snapshot = self.snapshot()
        for deployment_id in deployment_ids:
            state = snapshot.deployments.get(deployment_id)
            if state is not None and state.loaded and not state.active_request_ids:
                self.backends.stop(deployment_id, force=True)
                self.state_store.mutate(lambda current: self._mark_unloaded(current, deployment_id))

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

    def _mark_loaded(
        self,
        snapshot: RuntimeStateSnapshot,
        deployment: DeploymentConfig,
        pid: int,
    ) -> RuntimeStateSnapshot:
        snapshot = self._seed_snapshot(snapshot)
        runtime = snapshot.deployments[deployment.id]
        runtime.loaded = True
        runtime.loaded_at = utc_now_iso()
        runtime.process_id = pid
        runtime.resident_memory_fraction = deployment.memory_fraction
        return snapshot

    def _mark_unloaded(self, snapshot: RuntimeStateSnapshot, deployment_id: str) -> RuntimeStateSnapshot:
        runtime = snapshot.deployments[deployment_id]
        runtime.loaded = False
        runtime.process_id = None
        runtime.resident_memory_fraction = 0.0
        runtime.active_request_ids = []
        return snapshot

    def _mark_request_running(
        self,
        snapshot: RuntimeStateSnapshot,
        request_id: str,
        deployment: DeploymentConfig,
        auth: AuthResult,
    ) -> RuntimeStateSnapshot:
        snapshot = self._seed_snapshot(snapshot)
        queued = next(item for item in snapshot.queued_requests if item.id == request_id)
        queued.deployment_id = deployment.id
        queued.backend_runtime_id = deployment.backend_runtime_id
        queued.gpu_group_id = deployment.gpu_group_id
        queued.status = "running"
        queued.started_at = utc_now_iso()
        snapshot.active_requests[request_id] = queued
        snapshot.deployments[deployment.id].active_request_ids.append(request_id)
        snapshot.deployments[deployment.id].last_used_at = utc_now_iso()
        self.event_logger.audit(
            "request.started",
            auth.user_name,
            request_id=request_id,
            deployment_id=deployment.id,
            gpu_group_id=deployment.gpu_group_id,
        )
        return snapshot

    def _mark_request_finished(
        self,
        snapshot: RuntimeStateSnapshot,
        request_id: str,
        deployment_id: str,
    ) -> RuntimeStateSnapshot:
        request = snapshot.active_requests.pop(request_id)
        request.status = "completed"
        request.finished_at = utc_now_iso()
        runtime = snapshot.deployments[deployment_id]
        runtime.active_request_ids = [item for item in runtime.active_request_ids if item != request_id]
        runtime.last_used_at = utc_now_iso()
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
        request = snapshot.active_requests.pop(request_id)
        request.status = "failed"
        request.finished_at = utc_now_iso()
        request.error = error
        runtime = snapshot.deployments[deployment_id]
        runtime.active_request_ids = [item for item in runtime.active_request_ids if item != request_id]
        runtime.last_used_at = utc_now_iso()
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
            ),
            snapshot,
            utc_now(),
        )
        if not decision.accepted or decision.deployment_id is None:
            return
        deployment = self.config.deployments[decision.deployment_id]
        if decision.should_load:
            pid = self.backends.ensure_started(deployment)
            self.state_store.mutate(lambda state: self._mark_loaded(state, deployment, pid))
            await asyncio.sleep(0.5)
        adapter = self.backends.adapter_for(deployment.backend_runtime_id)
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
                    self.backends.stop(deployment_id, force=True)
                    self.state_store.mutate(lambda state: self._mark_unloaded(state, deployment_id))
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
            self.backends.stop(deployment_id, force=True)
            self.state_store.mutate(lambda state: self._mark_unloaded(state, deployment_id))
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
            "backend_health": snapshot.backend_health,
        }
