from __future__ import annotations

import json

from fastapi.testclient import TestClient
from shardon_admin_api.main import create_app as create_admin
from shardon_core.gpu.provider import MockGPUProvider
from shardon_core.services.runtime import ShardonRuntime
from shardon_core.state.models import ActiveRequest
from shardon_core.state.ownership import RouterInstance
from shardon_router_api import main as router
from test_api_surfaces import _make_repo_copy


def test_only_router_lifespan_recovers_dead_owners_and_preserves_live_work(tmp_path, monkeypatch):
    root = _make_repo_copy(tmp_path)
    monkeypatch.setenv("SHARDON_REPO_ROOT", str(root))
    runtime = ShardonRuntime(repo_root=root, gpu_provider=MockGPUProvider())
    live = RouterInstance(runtime.state_root)
    dead = RouterInstance(runtime.state_root)

    def seed(state):
        runtime._seed_snapshot(state)
        for label, owner in [("live", live), ("dead", dead)]:
            item = ActiveRequest(
                id=label,
                router_instance_id=owner.id,
                user_name="alice",
                api_key_id="k",
                deployment_id="chat-a",
                backend_runtime_id="mock",
                gpu_group_id="group",
                request_class="interactive",
                model_name="demo-chat",
                status="running",
                priority=1,
                created_at="2000-01-01T00:00:00+00:00",
                phase="in_flight",
            )
            state.active_requests[label] = item
            state.queued_requests.append(item)
            state.deployments["chat-a"].active_request_ids.append(label)

    runtime.state_store.mutate(seed)
    dead.close()

    async def no_backend_work(runtime):
        pass

    monkeypatch.setattr(router, "_run_background_tick", no_backend_work)
    try:
        with TestClient(create_admin()) as admin:
            assert admin.get("/health").status_code == 200
            assert admin.get("/runtime/activity").json()["in_flight"] == 2
        assert set(runtime.snapshot().active_requests) == {"live", "dead"}
        app = router.create_app()
        with TestClient(app) as client:
            assert client.get("/health").status_code == 200
            instance_id = app.state.runtime._router_instance.id
            assert not live.terminated(instance_id)
            state = runtime.snapshot()
            assert set(state.active_requests) == {"live"}
            assert [r.id for r in state.queued_requests] == ["live"]
            assert state.deployments["chat-a"].active_request_ids == ["live"]
        assert live.terminated(instance_id)
        saved = [
            json.loads(line)
            for path in runtime.usage_store.root.glob("*.jsonl")
            for line in path.read_text().splitlines()
        ]
        assert len(saved) == 1 and saved[0]["request_id"] == "dead"
        assert saved[0]["outcome"] == "failed" and saved[0]["interruption_reason"]
    finally:
        live.close()
