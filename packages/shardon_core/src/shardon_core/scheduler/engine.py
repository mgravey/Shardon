from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal
from urllib.parse import urlsplit

from shardon_core.config.schemas import DeploymentConfig, GPUGroupConfig, RepositoryConfig
from shardon_core.state.models import SLEEP_STATES, DeploymentRuntimeState, RuntimeStateSnapshot


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
    # Deployments to put to sleep (instead of stopping) before loading/waking.
    should_sleep: list[str] | None = None
    # The selected deployment is loaded but asleep and must be woken.
    should_wake: bool = False


@dataclass(slots=True)
class AdmissionPlan:
    """Memory actions needed before a deployment may become awake in a group."""

    stop: list[str] = field(default_factory=list)
    sleep: list[str] = field(default_factory=list)

    def actions(self) -> list[str]:
        return [*self.stop, *self.sleep]


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
            if snapshot.deployments.get(deployment.id, None)
            and snapshot.deployments[deployment.id].loaded
            and not self.unavailable_gpu_ids(
                self._deployment_selected_group(
                    deployment.id, snapshot, default=deployment.preferred_gpu_group_id()
                ),
                snapshot,
            )
        ]
        sleeping_candidates = [
            deployment
            for deployment in loaded_candidates
            if snapshot.deployments[deployment.id].state in SLEEP_STATES
        ]
        loaded_candidates = [
            deployment for deployment in loaded_candidates if deployment not in sleeping_candidates
        ]
        if request.request_class == "batch":
            loaded_batch = self._pick_best_loaded(loaded_candidates, snapshot)
            if loaded_batch is not None:
                loaded_group = self._deployment_selected_group(
                    loaded_batch.id,
                    snapshot,
                    default=loaded_batch.preferred_gpu_group_id(),
                )
                if (
                    request.target_gpu_group_id is None
                    or request.target_gpu_group_id == loaded_group
                ):
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

        wake_decision = self._schedule_wake(request, sleeping_candidates, snapshot, now)
        if wake_decision is not None:
            return wake_decision

        unavailable_groups: dict[str, list[str]] = {}
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
                missing = self.unavailable_gpu_ids(gpu_group_id, snapshot)
                if missing:
                    unavailable_groups[gpu_group_id] = missing
                    continue
                if self._group_is_draining(gpu_group_id, snapshot):
                    continue
                if (
                    deployment_state is not None
                    and deployment_state.loaded
                    and self._deployment_selected_group(
                        deployment.id, snapshot, default=gpu_group_id
                    )
                    == gpu_group_id
                ):
                    # Already loaded (awake or asleep) here: the reuse and wake
                    # paths above own that case; never restart it in place.
                    continue
                self_evict = self._self_evict_if_needed(deployment.id, gpu_group_id, snapshot)
                if (
                    self_evict
                    and deployment_state is not None
                    and deployment_state.active_request_ids
                ):
                    continue
                group = self.config.gpu_groups[gpu_group_id]
                plan = self.plan_admission(
                    group,
                    deployment,
                    snapshot,
                    self_evict=self_evict,
                    waking=False,
                )
                if plan is None:
                    continue
                if not self._plan_allowed(plan, deployment, gpu_group_id, request, snapshot, now):
                    continue
                return SchedulingDecision(
                    True,
                    deployment.id,
                    deployment.backend_runtime_id,
                    gpu_group_id,
                    200,
                    "deployment selected",
                    should_load=True,
                    should_evict=plan.stop,
                    should_sleep=plan.sleep,
                )
        reason = "no deployment currently admissible"
        if unavailable_groups:
            reason += "; unavailable GPUs (missing observations): " + "; ".join(
                f"{group_id}: {', '.join(gpu_ids)}"
                for group_id, gpu_ids in unavailable_groups.items()
            )
        return SchedulingDecision(False, None, None, None, 409, reason)

    def _schedule_wake(
        self,
        request: SchedulingRequest,
        sleeping_candidates: list[DeploymentConfig],
        snapshot: RuntimeStateSnapshot,
        now: datetime,
    ) -> SchedulingDecision | None:
        """Wake a sleeping deployment when its awake footprint is admissible.

        Waking needs the full footprint back, so it goes through the same
        admission planning as a cold load (minus the process start).
        """
        ordered = sorted(
            sleeping_candidates,
            key=lambda item: self._activity_key(snapshot.deployments[item.id]),
            reverse=True,
        )
        for deployment in ordered:
            gpu_group_id = self._deployment_selected_group(
                deployment.id, snapshot, default=deployment.preferred_gpu_group_id()
            )
            if request.target_gpu_group_id not in (None, gpu_group_id):
                continue
            if gpu_group_id not in self.config.gpu_groups:
                continue
            if self._group_is_draining(gpu_group_id, snapshot):
                continue
            plan = self.plan_admission(
                self.config.gpu_groups[gpu_group_id],
                deployment,
                snapshot,
                self_evict=[],
                waking=True,
            )
            if plan is None:
                continue
            if not self._plan_allowed(plan, deployment, gpu_group_id, request, snapshot, now):
                continue
            return SchedulingDecision(
                True,
                deployment.id,
                deployment.backend_runtime_id,
                gpu_group_id,
                200,
                "waking sleeping deployment",
                should_load=False,
                should_evict=plan.stop,
                should_sleep=plan.sleep,
                should_wake=True,
            )
        return None

    def _plan_allowed(
        self,
        plan: AdmissionPlan,
        deployment: DeploymentConfig,
        gpu_group_id: str,
        request: SchedulingRequest,
        snapshot: RuntimeStateSnapshot,
        now: datetime,
    ) -> bool:
        in_group = [
            deployment_id
            for deployment_id in plan.actions()
            if deployment_id != deployment.id
            and deployment_id in snapshot.deployments
            and snapshot.deployments[deployment_id].gpu_group_id == gpu_group_id
        ]
        if not in_group:
            return True
        awake_in_group = [
            deployment_id
            for deployment_id in in_group
            if snapshot.deployments[deployment_id].state not in SLEEP_STATES
        ]
        # Batches never displace an awake deployment.
        if request.request_class == "batch" and awake_in_group:
            return False
        return self._can_switch(
            gpu_group_id,
            request.priority,
            snapshot,
            now,
            bypass_grace_window=request.request_class == "manual",
        )

    def plan_admission(
        self,
        group: GPUGroupConfig,
        deployment: DeploymentConfig,
        snapshot: RuntimeStateSnapshot,
        *,
        self_evict: list[str],
        waking: bool,
    ) -> AdmissionPlan | None:
        """Return the stop/sleep actions that make ``deployment`` fit, or None.

        Static groups run one awake deployment at a time: every other awake
        deployment is stopped, or put to sleep when it opted into
        ``sleep_on_evict`` and does not share the incoming endpoint. Already
        sleeping deployments may stay when memory allows. Sleeping deployments
        are always evicted before awake ones.
        """
        others = [
            deployment_id
            for deployment_id in self._loaded_in_group(group.id, snapshot)
            if deployment_id != deployment.id
        ]
        target_endpoint = self.deployment_endpoint(deployment.id, group.id, snapshot, incoming=True)
        stop: list[str] = []
        sleep: list[str] = []
        for other in others:
            state = snapshot.deployments[other]
            if self.deployment_endpoint(other, group.id, snapshot) == target_endpoint:
                stop.append(other)
                continue
            if state.state in SLEEP_STATES:
                continue
            if self._can_sleep_on_evict(other, state):
                sleep.append(other)
            else:
                stop.append(other)
        busy = [
            deployment_id
            for deployment_id in [*stop, *sleep]
            if snapshot.deployments[deployment_id].active_request_ids
        ]
        if busy:
            return None
        while not self._group_allows_admission(
            group,
            deployment,
            snapshot,
            assumed_evictions=[*stop, *self_evict],
            sleep_evictions=sleep,
            waking=waking,
        ):
            # Free more: stop kept sleepers (least recently used first), then
            # deployments planned to sleep.
            kept_sleepers = [
                other
                for other in others
                if other not in stop
                and other not in sleep
                and snapshot.deployments[other].state == "sleeping"
                and not snapshot.deployments[other].active_request_ids
            ]
            pool = kept_sleepers or sleep
            if not pool:
                return None
            victim = min(pool, key=lambda item: self._activity_key(snapshot.deployments[item]))
            if victim in sleep:
                sleep.remove(victim)
            stop.append(victim)
        ordered_stop = [
            *[item for item in stop if snapshot.deployments[item].state in SLEEP_STATES],
            *[item for item in stop if snapshot.deployments[item].state not in SLEEP_STATES],
        ]
        return AdmissionPlan(
            stop=self._dedupe_ids([*ordered_stop, *self_evict]),
            sleep=sleep,
        )

    def deployment_endpoint(
        self,
        deployment_id: str,
        gpu_group_id: str,
        snapshot: RuntimeStateSnapshot,
        *,
        incoming: bool = False,
    ) -> str:
        """host:port a deployment serves on; two deployments sharing one cannot
        be co-resident (a sleeping process still holds its port)."""
        _ = snapshot, incoming
        deployment = self.config.deployments.get(deployment_id)
        backend = (
            self.config.backends.get(deployment.backend_runtime_id) if deployment else None
        )
        if backend is None:
            return f"unknown:{deployment_id}"
        base_url = backend.resolved_for_gpu_group(gpu_group_id).base_url
        return urlsplit(base_url).netloc.lower()

    def _can_sleep_on_evict(self, deployment_id: str, state: DeploymentRuntimeState) -> bool:
        policy = self.config.sleep_policy_for(deployment_id)
        return (
            policy.enabled
            and policy.sleep_on_evict
            and state.state == "ready"
            and not state.active_request_ids
        )

    @staticmethod
    def _activity_key(state: DeploymentRuntimeState) -> datetime:
        return state.last_activity_at() or datetime.min.replace(tzinfo=UTC)

    def unavailable_gpu_ids(self, gpu_group_id: str, snapshot: RuntimeStateSnapshot) -> list[str]:
        return [
            gpu_id
            for gpu_id in self.config.gpu_groups[gpu_group_id].gpu_ids
            if gpu_id not in snapshot.gpu_observations
        ]

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
        deployment_tasks = set(deployment.tasks)
        if request.task == "response" and "chat" in deployment_tasks:
            deployment_tasks.add("response")
        if request.task not in deployment_tasks:
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
            "response": backend.capabilities.responses,
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
                len(
                    snapshot.deployments.get(item.id, None).active_request_ids
                    if snapshot.deployments.get(item.id)
                    else []
                ),
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
        sleep_evictions: list[str] | None = None,
        waking: bool = False,
    ) -> bool:
        observation_totals = [snapshot.gpu_observations.get(gpu_id) for gpu_id in group.gpu_ids]
        if not observation_totals:
            return True
        deployment_memory_fraction = self.awake_footprint(deployment, group.id, snapshot)
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
        for deployment_id in sleep_evictions or []:
            state = snapshot.deployments.get(deployment_id)
            if state is None or state.gpu_group_id != group.id:
                continue
            residual = self.config.sleep_policy_for(deployment_id).sleeping_memory_fraction
            reclaimable_fraction += max(0.0, state.resident_memory_fraction - residual)
        own_state = snapshot.deployments.get(deployment.id)
        if waking and own_state is not None and own_state.gpu_group_id == group.id:
            # The sleeping residual is replaced by the awake footprint.
            reclaimable_fraction += own_state.resident_memory_fraction
        effective_reserved = max(0.0, currently_reserved - reclaimable_fraction)
        if effective_reserved + deployment_memory_fraction > memory_limit:
            return False
        for observation in observation_totals:
            if observation is None:
                return False
            free_fraction = observation.free_memory_mb / max(observation.total_memory_mb, 1)
            effective_free_fraction = min(1.0, free_fraction + reclaimable_fraction)
            if effective_free_fraction < (1 - deployment_memory_fraction) * 0.5:
                return False
        return True

    def awake_footprint(
        self,
        deployment: DeploymentConfig,
        gpu_group_id: str,
        snapshot: RuntimeStateSnapshot,
    ) -> float:
        """Memory fraction the deployment needs once awake in the group."""
        state = snapshot.deployments.get(deployment.id)
        if (
            state is not None
            and state.loaded
            and state.state in SLEEP_STATES
            and state.awake_memory_fraction is not None
        ):
            return state.awake_memory_fraction
        return deployment.memory_fraction_for_group(gpu_group_id)

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
        lower_or_equal = [
            request for request in queued_for_group if request.priority <= incoming_priority
        ]
        return bool(lower_or_equal)
