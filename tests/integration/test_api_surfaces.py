import inspect
import shutil
from pathlib import Path

from fastapi.params import Depends as DependsParam
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from shardon_core.auth.service import APIKeyService
from shardon_core.backends.base import BackendBinaryResponse
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
        if (
            isinstance(candidate, APIRoute)
            and candidate.path == path
            and method in candidate.methods
        ):
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
            "model_capabilities": ["text", "audio"],
            "runtime_launch_args_by_backend_type": {"vllm": ["--enable-prefix-caching"]},
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
    assert resources.json()["models"]["new-model"]["model_capabilities"] == ["text", "audio"]
    assert resources.json()["models"]["new-model"]["runtime_launch_args_by_backend_type"] == {
        "vllm": ["--enable-prefix-caching"]
    }

    models = router_client.get("/v1/models", headers={"Authorization": f"Bearer {secret}"})
    _assert_no_missing_query_dependencies(models, {"auth", "runtime"})
    assert models.status_code == 200
    payload = models.json()
    assert payload["object"] == "list"
    assert any(item["id"] == "demo-chat" for item in payload["data"])
    assert any(
        item["id"] == "demo-chat" and item["model_capabilities"] == ["text"]
        for item in payload["data"]
    )

    chat_without_auth = router_client.post(
        "/v1/chat/completions",
        json={"model": "demo-chat", "messages": [{"role": "user", "content": "ping"}]},
    )
    _assert_no_missing_query_dependencies(chat_without_auth, {"auth", "runtime"})
    assert chat_without_auth.status_code == 401


def test_router_responses_route_forwards_json_payload(tmp_path: Path, monkeypatch) -> None:
    repo_root = _make_repo_copy(tmp_path)
    monkeypatch.setenv("SHARDON_REPO_ROOT", str(repo_root))

    from shardon_router_api.main import create_app as create_router_app

    api_keys = APIKeyService(repo_root / "state", EventLogger(repo_root / "state"))
    _, secret = api_keys.create_key(
        key_id="responses-key",
        user_name="responses-user",
        priority=100,
        permissions=["inference"],
        actor="admin",
    )
    app = create_router_app()
    runtime = app.state.runtime
    client = TestClient(app)

    async def fake_response(payload, auth):  # type: ignore[no-untyped-def]
        _ = auth
        assert payload.model == "demo-chat"
        assert payload.input == "ping"
        dumped = payload.model_dump(mode="json")
        assert dumped["instructions"] == "be concise"
        return {"id": "resp-test", "object": "response", "model": payload.model}

    runtime.route_response = fake_response  # type: ignore[method-assign]

    unauthorized = client.post(
        "/v1/responses",
        json={"model": "demo-chat", "input": "ping"},
    )
    _assert_no_missing_query_dependencies(unauthorized, {"auth", "runtime"})
    assert unauthorized.status_code == 401

    response = client.post(
        "/v1/responses",
        headers={"Authorization": f"Bearer {secret}"},
        json={"model": "demo-chat", "input": "ping", "instructions": "be concise"},
    )
    assert response.status_code == 200
    assert response.json()["id"] == "resp-test"


def test_router_responses_route_streams_backend_events(tmp_path: Path, monkeypatch) -> None:
    repo_root = _make_repo_copy(tmp_path)
    monkeypatch.setenv("SHARDON_REPO_ROOT", str(repo_root))

    from shardon_router_api.main import create_app as create_router_app

    api_keys = APIKeyService(repo_root / "state", EventLogger(repo_root / "state"))
    _, secret = api_keys.create_key(
        key_id="responses-stream-key",
        user_name="responses-stream-user",
        priority=100,
        permissions=["inference"],
        actor="admin",
    )
    app = create_router_app()
    runtime = app.state.runtime
    client = TestClient(app)

    async def fake_stream_response(payload, auth):  # type: ignore[no-untyped-def]
        _ = auth
        assert payload.model == "demo-chat"
        assert payload.input == "ping"
        assert payload.model_dump(mode="json")["stream"] is True
        yield b'data: {"type":"response.output_text.delta","delta":"hel"}\n\n'
        yield b'data: {"type":"response.output_text.delta","delta":"lo"}\n\n'
        yield b"data: [DONE]\n\n"

    runtime.stream_response = fake_stream_response  # type: ignore[method-assign]

    with client.stream(
        "POST",
        "/v1/responses",
        headers={"Authorization": f"Bearer {secret}"},
        json={"model": "demo-chat", "input": "ping", "stream": True},
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = b"".join(response.iter_bytes())

    assert b'"delta":"hel"' in body
    assert body.endswith(b"data: [DONE]\n\n")


def test_router_completions_route_streams_backend_events(tmp_path: Path, monkeypatch) -> None:
    repo_root = _make_repo_copy(tmp_path)
    monkeypatch.setenv("SHARDON_REPO_ROOT", str(repo_root))

    from shardon_router_api.main import create_app as create_router_app

    api_keys = APIKeyService(repo_root / "state", EventLogger(repo_root / "state"))
    _, secret = api_keys.create_key(
        key_id="completions-stream-key",
        user_name="completions-stream-user",
        priority=100,
        permissions=["inference"],
        actor="admin",
    )
    app = create_router_app()
    runtime = app.state.runtime
    client = TestClient(app)

    async def fake_stream_completion(payload, auth):  # type: ignore[no-untyped-def]
        _ = auth
        assert payload.model == "demo-chat"
        assert payload.prompt == "ping"
        dumped = payload.model_dump(mode="json")
        assert dumped["stream"] is True
        assert dumped["suffix"] == "pong"
        yield b'data: {"choices":[{"text":"hel"}]}\n\n'
        yield b'data: {"choices":[{"text":"lo"}]}\n\n'
        yield b"data: [DONE]\n\n"

    runtime.stream_completion = fake_stream_completion  # type: ignore[method-assign]

    with client.stream(
        "POST",
        "/v1/completions",
        headers={"Authorization": f"Bearer {secret}"},
        json={"model": "demo-chat", "prompt": "ping", "stream": True, "suffix": "pong"},
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = b"".join(response.iter_bytes())

    assert b'"text":"hel"' in body
    assert body.endswith(b"data: [DONE]\n\n")


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
    _assert_dep_param(_route(router_app, "/v1/responses", "POST"), "auth")
    _assert_dep_param(_route(router_app, "/v1/responses", "POST"), "runtime")
    _assert_dep_param(_route(router_app, "/v1/completions", "POST"), "auth")
    _assert_dep_param(_route(router_app, "/v1/completions", "POST"), "runtime")
    _assert_dep_param(_route(router_app, "/v1/audio/speech", "POST"), "auth")
    _assert_dep_param(_route(router_app, "/v1/audio/speech", "POST"), "runtime")
    _assert_dep_param(_route(router_app, "/v1/audio/transcriptions", "POST"), "auth")
    _assert_dep_param(_route(router_app, "/v1/audio/transcriptions", "POST"), "runtime")
    _assert_dep_param(_route(router_app, "/v1/audio/translations", "POST"), "auth")
    _assert_dep_param(_route(router_app, "/v1/audio/translations", "POST"), "runtime")


def test_router_audio_routes_support_json_and_multipart(tmp_path: Path, monkeypatch) -> None:
    repo_root = _make_repo_copy(tmp_path)
    monkeypatch.setenv("SHARDON_REPO_ROOT", str(repo_root))

    from shardon_router_api.main import create_app as create_router_app

    api_keys = APIKeyService(repo_root / "state", EventLogger(repo_root / "state"))
    _, secret = api_keys.create_key(
        key_id="audio-key",
        user_name="audio-user",
        priority=100,
        permissions=["inference"],
        actor="admin",
    )
    app = create_router_app()
    runtime = app.state.runtime
    client = TestClient(app)

    async def fake_speech(payload, auth):  # type: ignore[no-untyped-def]
        _ = auth
        assert payload.model == "audio-demo"
        assert payload.input == "say hi"
        return BackendBinaryResponse(body=b"audio-bytes", content_type="audio/mpeg")

    async def fake_transcription(payload, uploaded_file, auth):  # type: ignore[no-untyped-def]
        _ = auth
        assert payload.model == "audio-demo"
        assert uploaded_file.filename == "sample.wav"
        return {"text": "transcribed"}

    async def fake_translation(payload, uploaded_file, auth):  # type: ignore[no-untyped-def]
        _ = auth
        assert payload.model == "audio-demo"
        assert uploaded_file.filename == "sample.wav"
        return {"text": "translated"}

    runtime.route_audio_speech = fake_speech  # type: ignore[method-assign]
    runtime.route_audio_transcription = fake_transcription  # type: ignore[method-assign]
    runtime.route_audio_translation = fake_translation  # type: ignore[method-assign]

    unauthorized = client.post(
        "/v1/audio/speech",
        json={"model": "audio-demo", "input": "say hi", "voice": "alloy"},
    )
    _assert_no_missing_query_dependencies(unauthorized, {"auth", "runtime"})
    assert unauthorized.status_code == 401

    speech = client.post(
        "/v1/audio/speech",
        headers={"Authorization": f"Bearer {secret}"},
        json={"model": "audio-demo", "input": "say hi", "voice": "alloy"},
    )
    assert speech.status_code == 200
    assert speech.content == b"audio-bytes"
    assert speech.headers["content-type"].startswith("audio/mpeg")

    transcription = client.post(
        "/v1/audio/transcriptions",
        headers={"Authorization": f"Bearer {secret}"},
        data={"model": "audio-demo"},
        files={"file": ("sample.wav", b"abc", "audio/wav")},
    )
    assert transcription.status_code == 200
    assert transcription.json()["text"] == "transcribed"

    translation = client.post(
        "/v1/audio/translations",
        headers={"Authorization": f"Bearer {secret}"},
        data={"model": "audio-demo"},
        files={"file": ("sample.wav", b"abc", "audio/wav")},
    )
    assert translation.status_code == 200
    assert translation.json()["text"] == "translated"

    transcription_text = client.post(
        "/v1/audio/transcriptions",
        headers={"Authorization": f"Bearer {secret}"},
        data={"model": "audio-demo", "response_format": "text"},
        files={"file": ("sample.wav", b"abc", "audio/wav")},
    )
    assert transcription_text.status_code == 200
    assert transcription_text.text == "transcribed"
    assert transcription_text.headers["content-type"].startswith("text/plain")


def test_chat_route_forwards_output_and_reasoning_options(tmp_path: Path, monkeypatch) -> None:
    repo_root = _make_repo_copy(tmp_path)
    monkeypatch.setenv("SHARDON_REPO_ROOT", str(repo_root))
    from shardon_router_api.main import create_app as create_router_app

    api_keys = APIKeyService(repo_root / "state", EventLogger(repo_root / "state"))
    _, secret = api_keys.create_key(
        key_id="chat-options", user_name="alice", priority=100,
        permissions=["inference"], actor="admin",
    )
    app = create_router_app()
    captured = []

    async def capture(task, model_name, payload, auth, **kwargs):
        assert task == "chat"
        captured.append(payload)
        return {"id": "chat-options-ok"}

    app.state.runtime._route_interactive = capture
    client = TestClient(app)
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": "answer", "schema": {"type": "object"}},
    }
    for effort in ["none", "low", "medium", "high", "xhigh"]:
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {secret}"},
            json={
                "model": "demo-chat",
                "messages": [{"role": "user", "content": "ping"}],
                "response_format": response_format,
                "reasoning_effort": effort,
            },
        )
        assert response.status_code == 200, response.text
        assert captured[-1]["response_format"] == response_format
        assert captured[-1]["reasoning_effort"] == effort


def test_activity_and_usage_are_public_filtered_and_read_only(tmp_path, monkeypatch):
    from shardon_admin_api.main import create_app
    from shardon_core.services.observability import TokenUsage
    from shardon_core.state.models import ActiveRequest
    from shardon_core.utils.time import utc_now_iso

    repo_root = _make_repo_copy(tmp_path)
    monkeypatch.setenv("SHARDON_REPO_ROOT", str(repo_root))
    app = create_app()
    client = TestClient(app)
    runtime = app.state.runtime
    token = client.post("/auth/login", json={"username": "admin", "password": "admin"}).json()[
        "access_token"
    ]
    headers = {"Authorization": f"Bearer {token}"}
    item = ActiveRequest(
        id="test-activity",
        api_key_id="old-key",
        user_name="alice",
        model_name="demo-chat",
        deployment_id="",
        backend_runtime_id="",
        gpu_group_id="",
        request_class="interactive",
        status="queued",
        priority=100,
        created_at=utc_now_iso(),
    )
    runtime.state_store.mutate(lambda s: s.model_copy(update={"queued_requests": [item]}))
    runtime.usage_store.record(item, "completed", TokenUsage(total_tokens=10))
    _, api_secret = runtime.api_keys.create_key(
        key_id="inference-key",
        user_name="bob",
        priority=100,
        permissions=["inference"],
        actor="admin",
    )

    def forbidden_side_effect(*args, **kwargs):
        raise AssertionError("read-only endpoint attempted a mutation or health check")

    for name in (
        "refresh_gpu_observations",
        "enforce_keep_free",
        "refresh_backend_health",
        "snapshot",
    ):
        monkeypatch.setattr(runtime, name, forbidden_side_effect)
    monkeypatch.setattr(runtime.state_store, "mutate", forbidden_side_effect)
    before = runtime.state_store.snapshot_path.read_bytes()
    # Public observability must not grant configuration or runtime control access.
    for method, path in (
        ("GET", "/resources"),
        ("GET", "/api-keys"),
        ("GET", "/runtime/status"),
        ("GET", "/runtime/environment"),
        ("POST", "/runtime/load/chat-a"),
        ("POST", "/runtime/unload/chat-a"),
        ("DELETE", "/resources/models/chat"),
    ):
        assert client.request(method, path).status_code == 401
        assert client.request(
            method, path, headers={"Authorization": f"Bearer {api_secret}"}
        ).status_code == 401
    for path in ("/runtime/activity", "/runtime/usage"):
        assert client.get(path).status_code == 200
        assert (
            client.get(path, headers={"Authorization": f"Bearer {api_secret}"}).status_code == 200
        )
        assert client.get(path, headers=headers).status_code == 200
    activity = client.get(
        "/runtime/activity?api_key=old-key&user=alice&model=demo-chat"
    ).json()
    assert activity["waiting"] == 1
    assert client.get("/runtime/activity?user=bob").json()["waiting"] == 0
    usage = client.get("/runtime/usage?group_by=user&api_key=old-key").json()
    assert usage["rows"][0]["identity"] == "alice"
    assert usage["totals"]["total_tokens"] == 10
    assert client.get("/runtime/usage?period=bad").status_code == 422
    assert client.get("/runtime/usage?group_by=bad").status_code == 422
    requested = client.get(
        "/runtime/usage?model_view=requested&model=demo-chat",
    ).json()
    assert requested["model_view"] == "requested"
    assert requested["totals"]["total_tokens"] == 10
    assert client.get("/runtime/usage?model_view=invalid").status_code == 422
    assert runtime.state_store.snapshot_path.read_bytes() == before


def test_admin_can_read_failed_backend_diagnostics_and_logs(tmp_path: Path, monkeypatch) -> None:
    import sys

    from shardon_admin_api.main import create_app

    monkeypatch.setenv("SHARDON_REPO_ROOT", str(_make_repo_copy(tmp_path)))
    app = create_app()
    runtime = app.state.runtime
    backend = runtime.config.backends[runtime.config.deployments["chat-a"].backend_runtime_id]
    backend.working_directory = str(tmp_path)
    backend.launch_command = [
        sys.executable, "-c",
        "import sys; print('synthetic GPU unavailable', flush=True); sys.exit(17)",
    ]
    backend.startup_timeout_seconds = 3
    backend.readiness_poll_interval_seconds = 0.01

    async def unavailable_health(*args, **kwargs):
        raise ConnectionError("synthetic backend unavailable")

    monkeypatch.setattr(runtime.backends, "health", unavailable_health)
    client = TestClient(app)
    login = client.post("/auth/login", json={"username": "admin", "password": "admin"})
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
    response = client.post("/runtime/load/chat-a", headers=headers)
    assert response.status_code == 409
    assert response.json()["detail"]["readiness"]["returncode"] == 17
    logs = client.get("/runtime/logs/chat-a", headers=headers)
    assert logs.status_code == 200
    assert logs.json()["lines"] == ["synthetic GPU unavailable"]
    events = client.get("/runtime/events", headers=headers)
    assert events.status_code == 200
    assert any('"category": "backend.readiness_failed"' in line for line in events.json()["lines"])
    assert client.get("/runtime/logs/chat-a").status_code == 401
