from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from shardon_core.config.schemas import DeploymentConfig, GPUGroupConfig, RepositoryConfig
from shardon_core.state.models import ActiveRequest, RuntimeStateSnapshot


@dataclass(slots=True)
class SchedulingDecision:
    accepted: bool
    deployment_id: str | None
    backend_runtime_id: str | None
    gpu_group_id: str | None
    status_code: int
    reason: str
    should_load: bool = False
    should_evict: list[str] | None = None


@dataclass(slots=True)
class SchedulingRequest:
    model_name: str
    task: str
    priority: int
    request_class: str
    request_id: str
    deployment_id: str | None = None
    target_gpu_group_id: str | None = None
    required_capability: Literal["text", "audio", "image", "video"] | None = None


class SchedulerEngine:
    def __init__(self, config: RepositoryConfig) -> None:
        self.config = config
        self.grace_window_seconds = config.global_config.switch_grace_window_seconds

    def schedule(
        self,
        request: SchedulingRequest,
        snapshot: RuntimeStateSnapshot,
        now: datetime,
    ) -> SchedulingDecision:
        candidates = self._matching_deployments(request)
        if not candidates:
            return SchedulingDecision(False, None, None, None, 404, "no compatible deployment")

        loaded_candidates = [
            deployment
            for deployment in candidates
            if snapshot.deployments.get(deployment.id, None) and snapshot.deployments[deployment.id].loaded
        ]
        if request.request_class == "batch":
            loaded_batch = self._pick_best_loaded(loaded_candidates, snapshot)
            if loaded_batch is not None:
                loaded_group = self._deployment_selected_group(
                    loaded_batch.id,
                    snapshot,
                    default=loaded_batch.preferred_gpu_group_id(),
                )
                if request.target_gpu_group_id is None or request.target_gpu_group_id == loaded_group:
                    return SchedulingDecision(
                        True,
                        loaded_batch.id,
                        loaded_batch.backend_runtime_id,
                        loaded_group,
                        200,
                        "batch scheduled on loaded deployment",
                    )
        else:
            loaded = self._pick_best_loaded(loaded_candidates, snapshot)
            if loaded is not None:
                loaded_group = self._deployment_selected_group(
                    loaded.id,
                    snapshot,
                    default=loaded.preferred_gpu_group_id(),
                )
                if self._group_is_draining(loaded_group, snapshot):
                    loaded = None
            if loaded is not None and (
                request.target_gpu_group_id is None or request.target_gpu_group_id == loaded_group
            ):
                return SchedulingDecision(
                    True,
                    loaded.id,
                    loaded.backend_runtime_id,
                    loaded_group,
                    200,
                    "reusing loaded deployment",
                )

        for deployment in candidates:
            deployment_state = snapshot.deployments.get(deployment.id)
            group_candidates = (
                [request.target_gpu_group_id]
                if request.target_gpu_group_id is not None
                else deployment.eligible_gpu_group_ids()
            )
            for gpu_group_id in group_candidates:
                if gpu_group_id is None:
                    continue
                if gpu_group_id not in deployment.eligible_gpu_group_ids():
                    continue
                if gpu_group_id not in self.config.gpu_groups:
                    continue
                if self._group_is_draining(gpu_group_id, snapshot):
                    continue
                loaded_here = self._loaded_in_group(gpu_group_id, snapshot)
                self_evict = self._self_evict_if_needed(deployment.id, gpu_group_id, snapshot)
                if self_evict and deployment_state is not None and deployment_state.active_request_ids:
                    continue
                assumed_evictions = self._dedupe_ids([*loaded_here, *self_evict])
                busy_evictions = [
                    deployment_id
                    for deployment_id in assumed_evictions
                    if deployment_id != deployment.id
                    and snapshot.deployments.get(deployment_id, None)
                    and snapshot.deployments[deployment_id].active_request_ids
                ]
                if busy_evictions:
                    continue
                can_switch = (
                    not loaded_here
                    or self._can_switch(
                        gpu_group_id,
                        request.priority,
                        snapshot,
                        now,
                        bypass_grace_window=request.request_class == "manual",
                    )
                )
                if request.request_class == "batch" and loaded_here:
                    continue
                if loaded_here and not can_switch:
                    continue
                group = self.config.gpu_groups[gpu_group_id]
                if not self._group_allows_admission(
                    group,
                    deployment,
                    snapshot,
                    assumed_evictions=assumed_evictions,
                ):
                    continue
                return SchedulingDecision(
                    True,
                    deployment.id,
                    deployment.backend_runtime_id,
                    gpu_group_id,
                    200,
                    "deployment selected",
                    should_load=True,
                    should_evict=assumed_evictions,
                )
        return SchedulingDecision(False, None, None, None, 409, "no deployment currently admissible")

    def _matching_deployments(self, request: SchedulingRequest) -> list[DeploymentConfig]:
        if request.deployment_id is not None:
            deployment = self.config.deployments.get(request.deployment_id)
            if deployment is None or not deployment.enabled:
                return []
            return [deployment]
        return [
            deployment
            for deployment in self.config.deployments.values()
            if deployment.enabled
            and deployment.api_model_name == request.model_name
            and self._deployment_supports_request(deployment, request)
        ]

    def _deployment_supports_request(
        self,
        deployment: DeploymentConfig,
        request: SchedulingRequest,
    ) -> bool:
        if request.task not in deployment.tasks:
            return False
        model = self.config.models.get(deployment.model_id)
        backend = self.config.backends.get(deployment.backend_runtime_id)
        if model is None or backend is None:
            return False
        if request.required_capability is not None:
            if request.required_capability not in model.model_capabilities:
                return False
            if (
                deployment.deployment_capabilities
                and request.required_capability not in deployment.deployment_capabilities
            ):
                return False
            if request.required_capability not in backend.capabilities.modalities:
                return False
        capability_map = {
            "chat": backend.capabilities.chat,
            "completion": backend.capabilities.completions,
            "embedding": backend.capabilities.embeddings,
            "audio_speech": backend.capabilities.audio_speech,
            "audio_transcription": backend.capabilities.audio_transcriptions,
            "audio_translation": backend.capabilities.audio_translations,
        }
        return capability_map.get(request.task, True)

    def _pick_best_loaded(
        self,
        candidates: list[DeploymentConfig],
        snapshot: RuntimeStateSnapshot,
    ) -> DeploymentConfig | None:
        if not candidates:
            return None
        return sorted(
            candidates,
            key=lambda item: (
                len(snapshot.deployments.get(item.id, None).active_request_ids if snapshot.deployments.get(item.id) else []),
                snapshot.deployments.get(item.id).last_used_at or "",
            ),
        )[0]

    def _loaded_in_group(self, gpu_group_id: str, snapshot: RuntimeStateSnapshot) -> list[str]:
        return [
            deployment_id
            for deployment_id, state in snapshot.deployments.items()
            if state.gpu_group_id == gpu_group_id and state.loaded
        ]

    def _deployment_selected_group(
        self,
        deployment_id: str,
        snapshot: RuntimeStateSnapshot,
        *,
        default: str,
    ) -> str:
        state = snapshot.deployments.get(deployment_id)
        if state is None:
            return default
        return state.selected_gpu_group_id or state.gpu_group_id or default

    def _self_evict_if_needed(
        self,
        deployment_id: str,
        target_gpu_group_id: str,
        snapshot: RuntimeStateSnapshot,
    ) -> list[str]:
        state = snapshot.deployments.get(deployment_id)
        if state is None or not state.loaded:
            return []
        selected_gpu_group_id = state.selected_gpu_group_id or state.gpu_group_id
        if selected_gpu_group_id == target_gpu_group_id:
            return []
        return [deployment_id]

    def _dedupe_ids(self, values: list[str]) -> list[str]:
        deduped: list[str] = []
        for value in values:
            if value not in deduped:
                deduped.append(value)
        return deduped

    def _group_is_draining(self, gpu_group_id: str, snapshot: RuntimeStateSnapshot) -> bool:
        drain = snapshot.drains.get(gpu_group_id)
        return drain is not None and drain.status == "pending"

    def _group_allows_admission(
        self,
        group: GPUGroupConfig,
        deployment: DeploymentConfig,
        snapshot: RuntimeStateSnapshot,
        *,
        assumed_evictions: list[str],
    ) -> bool:
        observation_totals = [snapshot.gpu_observations.get(gpu_id) for gpu_id in group.gpu_ids]
        if not observation_totals:
            return True
        deployment_memory_fraction = deployment.memory_fraction_for_group(group.id)
        memory_limit = group.usable_memory_fraction
        currently_reserved = sum(
            state.resident_memory_fraction
            for state in snapshot.deployments.values()
            if state.gpu_group_id == group.id and state.loaded
        )
        reclaimable_fraction = sum(
            snapshot.deployments[deployment_id].resident_memory_fraction
            for deployment_id in assumed_evictions
            if deployment_id in snapshot.deployments
            and snapshot.deployments[deployment_id].gpu_group_id == group.id
        )
        effective_reserved = max(0.0, currently_reserved - reclaimable_fraction)
        if effective_reserved + deployment_memory_fraction > memory_limit:
            return False
        for observation in observation_totals:
            if observation is None:
                continue
            free_fraction = observation.free_memory_mb / max(observation.total_memory_mb, 1)
            effective_free_fraction = min(1.0, free_fraction + reclaimable_fraction)
            if effective_free_fraction < (1 - deployment_memory_fraction) * 0.5:
                return False
        return True

    def _can_switch(
        self,
        gpu_group_id: str,
        incoming_priority: int,
        snapshot: RuntimeStateSnapshot,
        now: datetime,
        *,
        bypass_grace_window: bool = False,
    ) -> bool:
        if bypass_grace_window:
            return True
        loaded_model_names = {
            self.config.deployments[deployment_id].api_model_name
            for deployment_id in self._loaded_in_group(gpu_group_id, snapshot)
            if deployment_id in self.config.deployments
        }
        queued_for_group = [
            request
            for request in snapshot.queued_requests
            if request.request_class == "interactive" and request.model_name in loaded_model_names
        ]
        if not queued_for_group:
            return True
        oldest = min(
            queued_for_group,
            key=lambda request: request.created_at,
        )
        oldest_timestamp = datetime.fromisoformat(oldest.created_at)
        if (now - oldest_timestamp).total_seconds() < self.grace_window_seconds:
            return False
        lower_or_equal = [request for request in queued_for_group if request.priority <= incoming_priority]
        return bool(lower_or_equal)
