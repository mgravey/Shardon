from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

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
        candidates = [
            deployment
            for deployment in self.config.deployments.values()
            if deployment.enabled
            and deployment.api_model_name == request.model_name
            and request.task in deployment.tasks
        ]
        if not candidates:
            return SchedulingDecision(False, None, None, None, 404, "no compatible deployment")

        loaded_candidates = [
            deployment for deployment in candidates if snapshot.deployments.get(deployment.id, None) and
            snapshot.deployments[deployment.id].loaded
        ]
        if request.request_class == "batch":
            loaded_batch = self._pick_best_loaded(loaded_candidates, snapshot)
            if loaded_batch is not None:
                return SchedulingDecision(
                    True,
                    loaded_batch.id,
                    loaded_batch.backend_runtime_id,
                    loaded_batch.gpu_group_id,
                    200,
                    "batch scheduled on loaded deployment",
                )
        else:
            loaded = self._pick_best_loaded(loaded_candidates, snapshot)
            if loaded is not None and not self._group_is_draining(loaded.gpu_group_id, snapshot):
                return SchedulingDecision(
                    True,
                    loaded.id,
                    loaded.backend_runtime_id,
                    loaded.gpu_group_id,
                    200,
                    "reusing loaded deployment",
                )

        for deployment in candidates:
            if self._group_is_draining(deployment.gpu_group_id, snapshot):
                continue
            group = self.config.gpu_groups[deployment.gpu_group_id]
            loaded_here = self._loaded_in_group(group.id, snapshot)
            can_switch = not loaded_here or self._can_switch(group.id, request.priority, snapshot, now)
            if request.request_class == "batch" and loaded_here:
                continue
            if loaded_here and not can_switch:
                continue
            if not self._group_allows_admission(
                group,
                deployment,
                snapshot,
                assumed_evictions=loaded_here if can_switch else [],
            ):
                continue
            return SchedulingDecision(
                True,
                deployment.id,
                deployment.backend_runtime_id,
                deployment.gpu_group_id,
                200,
                "deployment selected",
                should_load=True,
                should_evict=loaded_here,
            )
        return SchedulingDecision(False, None, None, None, 409, "no deployment currently admissible")

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
        )
        effective_reserved = max(0.0, currently_reserved - reclaimable_fraction)
        if effective_reserved + deployment.memory_fraction > memory_limit:
            return False
        for observation in observation_totals:
            if observation is None:
                continue
            free_fraction = observation.free_memory_mb / max(observation.total_memory_mb, 1)
            effective_free_fraction = min(1.0, free_fraction + reclaimable_fraction)
            if effective_free_fraction < (1 - deployment.memory_fraction) * 0.5:
                return False
        return True

    def _can_switch(
        self,
        gpu_group_id: str,
        incoming_priority: int,
        snapshot: RuntimeStateSnapshot,
        now: datetime,
    ) -> bool:
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
