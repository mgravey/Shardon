from __future__ import annotations

from shardon_core.config.schemas import DeploymentConfig
from shardon_core.state.models import RuntimeStateSnapshot


def sleep_refusal(
    snapshot: RuntimeStateSnapshot,
    deployment: DeploymentConfig,
    *,
    require_idle: bool,
) -> str | None:
    """Why ``deployment`` cannot be put to sleep now, or None if it can.

    Shared by the runtime (which enforces it) and the public activity view
    (which only offers the Sleep button when this returns None).
    """
    runtime = snapshot.deployments.get(deployment.id)
    if runtime is None or not runtime.loaded:
        return "deployment is not loaded"
    if runtime.state == "sleeping":
        return "already sleeping"
    if runtime.state != "ready":
        return f"deployment is {runtime.state}"
    if runtime.active_request_ids:
        return "deployment is serving requests"
    if not require_idle:
        return None
    for request in snapshot.queued_requests:
        if request.status != "queued":
            continue
        if request.deployment_id == deployment.id or (
            not request.deployment_id and request.model_name == deployment.api_model_name
        ):
            return "requests are queued for this deployment"
    for job in snapshot.batch_jobs.values():
        if job.status not in {"queued", "running"}:
            continue
        if job.deployment_id == deployment.id or (
            job.deployment_id is None and job.model_name == deployment.api_model_name
        ):
            return "batch work targets this deployment"
    return None
