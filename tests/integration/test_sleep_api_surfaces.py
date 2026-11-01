from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from shardon_core.auth.service import APIKeyService
from shardon_core.logging.events import EventLogger

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages" / "shardon_core" / "tests"))

from sleep_fixtures import build_repo, deployment, mock_backend, mock_stats  # noqa: E402

DEV_ENDPOINTS = [
    ("POST", "/sleep?level=1&mode=abort"),
    ("POST", "/wake_up"),
    ("POST", "/wake_up?tags=weights"),
    ("POST", "/collective_rpc"),
    ("GET", "/is_sleeping"),
    ("POST", "/v1/sleep"),
    ("GET", "/v1/is_sleeping"),
    ("POST", "/reset_prefix_cache"),
]


@pytest.fixture
def repo(tmp_path, monkeypatch):
    root = build_repo(
        tmp_path,
        backends=[
            mock_backend("vllm-a", sleep={"enabled": True}),
            mock_backend("vllm-plain"),
        ],
        deployments=[deployment("dep-a", "vllm-a"), deployment("dep-plain", "vllm-plain")],
    )
    monkeypatch.setenv("SHARDON_REPO_ROOT", str(root))
    return root


def test_router_never_proxies_vllm_dev_endpoints(repo) -> None:
    from shardon_router_api.main import create_app

    _, secret = APIKeyService(repo / "state", EventLogger(repo / "state")).create_key(
        key_id="k", user_name="u", priority=100, permissions=["inference"], actor="admin"
    )
    client = TestClient(create_app())
    for method, path in DEV_ENDPOINTS:
        response = client.request(
            method,
            path,
            headers={"Authorization": f"Bearer {secret}"},
            json={"method": "reload_weights"},
        )
        assert response.status_code in {404, 405}, (method, path, response.status_code)


def test_admin_and_public_sleep_controls(repo) -> None:
    from shardon_admin_api.main import create_app

    app = create_app()
    runtime = app.state.runtime
    client = TestClient(app)
    try:
        # Admin endpoints require a token.
        assert client.post("/runtime/sleep/dep-a").status_code == 401
        assert client.post("/runtime/wake/dep-a").status_code == 401
        token = client.post(
            "/auth/login", json={"username": "admin", "password": "admin"}
        ).json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        assert client.post("/runtime/load/dep-a", headers=headers).status_code == 200

        # Public sleep: no admin token needed.
        public = client.post("/runtime/activity/deployments/dep-a/sleep")
        assert public.status_code == 200, public.text
        assert runtime.snapshot().deployments["dep-a"].state == "sleeping"
        assert mock_stats(runtime, "dep-a")["sleeping"] is True

        activity = client.get("/runtime/activity").json()
        rows = {d["id"]: d for m in activity["models"] for d in m["deployments"]}
        assert rows["dep-a"]["state"] == "sleeping"
        assert rows["dep-a"]["sleep_enabled"] is True
        assert rows["dep-plain"]["sleep_enabled"] is False

        status = client.get("/runtime/status", headers=headers).json()
        assert status["deployments"]["dep-a"]["state"] == "sleeping"
        assert status["sleep_policies"]["dep-a"]["idle_seconds"] == 600

        wake = client.post("/runtime/wake/dep-a", headers=headers)
        assert wake.status_code == 200, wake.text
        assert runtime.snapshot().deployments["dep-a"].state == "ready"

        admin_sleep = client.post("/runtime/sleep/dep-a", headers=headers)
        assert admin_sleep.status_code == 200

        # Stop works on a sleeping deployment (admin "Stop" button).
        stop = client.post("/runtime/unload/dep-a", headers=headers)
        assert stop.status_code == 200
        assert runtime.snapshot().deployments["dep-a"].loaded is False

        # Public sleep is refused for deployments without sleep enabled.
        refused = client.post("/runtime/activity/deployments/dep-plain/sleep")
        assert refused.status_code == 409
        unknown = client.post("/runtime/activity/deployments/nope/sleep")
        assert unknown.status_code == 404
    finally:
        for deployment_id in list(runtime.backends.supervisor.processes):
            runtime.backends.supervisor.kill(deployment_id)
