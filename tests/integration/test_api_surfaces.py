import os
import shutil
from pathlib import Path

from fastapi.testclient import TestClient

from shardon_core.auth.service import APIKeyService
from shardon_core.logging.events import EventLogger


def _make_repo_copy(tmp_path: Path) -> Path:
    source_root = Path(__file__).resolve().parents[2]
    target_root = tmp_path / "repo"
    shutil.copytree(source_root / "config", target_root / "config")
    (target_root / "state").mkdir(parents=True, exist_ok=True)
    return target_root


def test_admin_and_router_health_and_models(tmp_path: Path, monkeypatch) -> None:
    repo_root = _make_repo_copy(tmp_path)
    monkeypatch.setenv("SHARDON_REPO_ROOT", str(repo_root))
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")

    from shardon_admin_api.main import create_app as create_admin_app
    from shardon_router_api.main import create_app as create_router_app

    api_keys = APIKeyService(repo_root / "state", EventLogger(repo_root / "state"))
    _, secret = api_keys.create_key(
        key_id="demo-key",
        user_name="demo-user",
        priority=100,
        permissions=["inference"],
        actor="admin",
    )

    admin_client = TestClient(create_admin_app())
    router_client = TestClient(create_router_app())

    admin_health = admin_client.get("/health")
    router_health = router_client.get("/health")
    assert admin_health.status_code == 200
    assert router_health.status_code == 200

    login = admin_client.post("/auth/login", json={"username": "admin", "password": "admin"})
    assert login.status_code == 200
    admin_token = login.json()["access_token"]

    environment = admin_client.get(
        "/runtime/environment",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert environment.status_code == 200
    assert environment.json()["hf_token_configured"] is True

    onboard = admin_client.post(
        "/workflows/model-onboarding",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={
            "model_id": "new-model",
            "source": "org/new-model",
            "display_name": "New Model",
            "backend_compatibility": ["vllm"],
            "tasks": ["chat"],
            "create_deployment": True,
            "deployment_id": "new-model-a",
            "api_model_name": "new-model",
            "deployment_display_name": "New Model / Group A",
            "backend_runtime_id": "mock-vllm-v1",
            "gpu_group_id": "group-a",
            "memory_fraction": 0.7,
        },
    )
    assert onboard.status_code == 200

    resources = admin_client.get(
        "/resources",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert resources.status_code == 200
    assert "new-model" in resources.json()["models"]
    assert "new-model-a" in resources.json()["deployments"]

    models = router_client.get("/v1/models", headers={"Authorization": f"Bearer {secret}"})
    assert models.status_code == 200
    payload = models.json()
    assert payload["object"] == "list"
    assert any(item["id"] == "demo-chat" for item in payload["data"])
    assert any(item["id"] == "new-model" for item in payload["data"])
