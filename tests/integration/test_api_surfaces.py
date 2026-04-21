import inspect
import os
import shutil
from pathlib import Path

from fastapi.params import Depends as DependsParam
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from shardon_core.auth.service import APIKeyService
from shardon_core.logging.events import EventLogger


def _make_repo_copy(tmp_path: Path) -> Path:
    source_root = Path(__file__).resolve().parents[2]
    target_root = tmp_path / "repo"
    shutil.copytree(source_root / "config", target_root / "config")
    for directory_name in ("admins-available", "admins-enabled"):
        admin_dir = target_root / "config" / "auth" / directory_name
        for admin_user in admin_dir.glob("*.yaml"):
            lines = []
            for line in admin_user.read_text(encoding="utf-8").splitlines():
                if line.startswith("created_at: "):
                    created_at = line.removeprefix("created_at: ").strip()
                    line = f'created_at: "{created_at}"'
                lines.append(line)
            admin_user.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (target_root / "state").mkdir(parents=True, exist_ok=True)
    return target_root


def _assert_no_missing_query_dependencies(response, names: set[str]) -> None:
    if response.status_code != 422:
        return
    detail = response.json().get("detail", [])
    missing_query_names = {
        item["loc"][1]
        for item in detail
        if item.get("type") == "missing" and item.get("loc", [None, None])[0] == "query"
    }
    assert missing_query_names.isdisjoint(names), response.json()


def _route(app, path: str, method: str) -> APIRoute:
    for candidate in app.routes:
        if isinstance(candidate, APIRoute) and candidate.path == path and method in candidate.methods:
            return candidate
    raise AssertionError(f"route {method} {path} not found")


def _assert_dep_param(route: APIRoute, name: str) -> None:
    parameter = inspect.signature(route.endpoint).parameters[name]
    assert isinstance(parameter.default, DependsParam), (
        f"Expected direct Depends() default for {route.path} parameter '{name}'"
    )


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
    _assert_no_missing_query_dependencies(login, {"runtime"})
    assert login.status_code == 200
    admin_token = login.json()["access_token"]

    runtime_status = admin_client.get(
        "/runtime/status",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    _assert_no_missing_query_dependencies(runtime_status, {"admin_identity", "runtime"})
    assert runtime_status.status_code == 200

    environment = admin_client.get(
        "/runtime/environment",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    _assert_no_missing_query_dependencies(environment, {"admin_identity", "runtime"})
    assert environment.status_code == 200
    assert environment.json()["hf_token_configured"] is True

    load_attempt = admin_client.post(
        "/runtime/load/chat-a",
    )
    _assert_no_missing_query_dependencies(load_attempt, {"username", "runtime"})
    assert load_attempt.status_code == 401

    selector_load_attempt = admin_client.post(
        "/runtime/load",
        json={"deployment_id": "chat-a"},
    )
    _assert_no_missing_query_dependencies(selector_load_attempt, {"username", "runtime"})
    assert selector_load_attempt.status_code == 401

    unload_attempt = admin_client.post(
        "/runtime/unload/chat-a",
    )
    _assert_no_missing_query_dependencies(unload_attempt, {"username", "runtime"})
    assert unload_attempt.status_code == 401

    clear_queue = admin_client.post(
        "/runtime/queue/clear",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"interactive": True, "batches": True},
    )
    _assert_no_missing_query_dependencies(clear_queue, {"username", "runtime"})
    assert clear_queue.status_code == 200
    assert "cleared_interactive_requests" in clear_queue.json()
    assert "cancelled_batch_jobs" in clear_queue.json()

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
    _assert_no_missing_query_dependencies(models, {"auth", "runtime"})
    assert models.status_code == 200
    payload = models.json()
    assert payload["object"] == "list"
    assert any(item["id"] == "demo-chat" for item in payload["data"])

    chat_without_auth = router_client.post(
        "/v1/chat/completions",
        json={"model": "demo-chat", "messages": [{"role": "user", "content": "ping"}]},
    )
    _assert_no_missing_query_dependencies(chat_without_auth, {"auth", "runtime"})
    assert chat_without_auth.status_code == 401


def test_route_signatures_use_direct_depends(tmp_path: Path, monkeypatch) -> None:
    repo_root = _make_repo_copy(tmp_path)
    monkeypatch.setenv("SHARDON_REPO_ROOT", str(repo_root))

    from shardon_admin_api.main import create_app as create_admin_app
    from shardon_router_api.main import create_app as create_router_app

    admin_app = create_admin_app()
    router_app = create_router_app()

    _assert_dep_param(_route(admin_app, "/auth/login", "POST"), "runtime")
    _assert_dep_param(_route(admin_app, "/runtime/status", "GET"), "admin_identity")
    _assert_dep_param(_route(admin_app, "/runtime/status", "GET"), "runtime")
    _assert_dep_param(_route(admin_app, "/runtime/environment", "GET"), "admin_identity")
    _assert_dep_param(_route(admin_app, "/runtime/environment", "GET"), "runtime")
    _assert_dep_param(_route(admin_app, "/runtime/load/{deployment_id}", "POST"), "username")
    _assert_dep_param(_route(admin_app, "/runtime/load/{deployment_id}", "POST"), "runtime")
    _assert_dep_param(_route(admin_app, "/runtime/load", "POST"), "username")
    _assert_dep_param(_route(admin_app, "/runtime/load", "POST"), "runtime")
    _assert_dep_param(_route(admin_app, "/runtime/unload/{deployment_id}", "POST"), "username")
    _assert_dep_param(_route(admin_app, "/runtime/unload/{deployment_id}", "POST"), "runtime")
    _assert_dep_param(_route(admin_app, "/runtime/queue/clear", "POST"), "username")
    _assert_dep_param(_route(admin_app, "/runtime/queue/clear", "POST"), "runtime")
    _assert_dep_param(_route(router_app, "/v1/models", "GET"), "auth")
    _assert_dep_param(_route(router_app, "/v1/models", "GET"), "runtime")
    _assert_dep_param(_route(router_app, "/v1/chat/completions", "POST"), "auth")
    _assert_dep_param(_route(router_app, "/v1/chat/completions", "POST"), "runtime")
