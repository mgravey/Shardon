"""Fixtures for sleep-mode and elastic-memory tests.

They build a minimal repository config whose backends run the real
``demo/runtimes/mock-vllm-v2`` launcher (with ``--enable-sleep-mode``) as
separate processes on free ports, so Shardon drives the actual HTTP sleep API.
"""

from __future__ import annotations

import shutil
import socket
import sys
from pathlib import Path
from typing import Any

import httpx
import yaml
from shardon_core.auth.service import AuthResult
from shardon_core.gpu.provider import MockGPUProvider
from shardon_core.services.runtime import ShardonRuntime


def source_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "config").exists() and (parent / "demo").exists():
            return parent
    raise RuntimeError("repository root not found")


MOCK_LAUNCHER = source_repo_root() / "demo" / "runtimes" / "mock-vllm-v2" / "launcher.py"

AUTH = AuthResult(id="k1", user_name="alice", priority=100, permissions=["inference"])


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def mock_backend(
    backend_id: str,
    *,
    port: int | None = None,
    sleep: dict[str, Any] | None = None,
    environment: dict[str, str] | None = None,
    enable_sleep_mode: bool = True,
    **extra: Any,
) -> dict[str, Any]:
    port = port or free_port()
    command = [sys.executable, str(MOCK_LAUNCHER), "--runtime-label", backend_id]
    if "port_range" not in extra and not extra.get("per_deployment_ports"):
        # Without per-deployment ports the launcher listens on the fixed port;
        # otherwise it reads IGF_SHARDON_RUNTIME_PORT.
        command += ["--port", str(port)]
    if enable_sleep_mode:
        command.append("--enable-sleep-mode")
    payload: dict[str, Any] = {
        "id": backend_id,
        "backend_type": "vllm",
        "version": "0.30.0",
        "display_name": backend_id,
        "runtime_dir": str(MOCK_LAUNCHER.parent),
        "base_url": f"http://127.0.0.1:{port}",
        "launch_command": command,
        "health_path": "/v1/models",
        "environment": {"VLLM_SERVER_DEV_MODE": "1", **(environment or {})},
        "startup_timeout_seconds": 30,
        "stop_timeout_seconds": 5,
    }
    if sleep is not None:
        payload["sleep"] = sleep
    payload.update(extra)
    return payload


def deployment(
    deployment_id: str,
    backend_id: str,
    *,
    group: str = "rtx",
    api_model_name: str | None = None,
    memory_fraction: float = 0.45,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "id": deployment_id,
        "model_id": "demo-chat-model",
        "backend_runtime_id": backend_id,
        "gpu_group_id": group,
        "api_model_name": api_model_name or f"model-{deployment_id}",
        "display_name": deployment_id,
        "memory_fraction": memory_fraction,
        "tasks": ["chat", "response", "completion"],
        **extra,
    }


def build_repo(
    tmp_path: Path,
    *,
    backends: list[dict[str, Any]],
    deployments: list[dict[str, Any]],
    group: dict[str, Any] | None = None,
    global_overrides: dict[str, Any] | None = None,
) -> Path:
    source = source_repo_root()
    root = tmp_path / "repo"
    config = root / "config"
    (config / "auth").mkdir(parents=True)
    shutil.copytree(source / "config" / "auth", config / "auth", dirs_exist_ok=True)
    for directory_name in ("admins-available", "admins-enabled"):
        for admin_user in (config / "auth" / directory_name).glob("*.yaml"):
            payload = yaml.safe_load(admin_user.read_text(encoding="utf-8"))
            payload["created_at"] = str(payload["created_at"])
            admin_user.write_text(yaml.safe_dump(payload), encoding="utf-8")
    global_config = {
        "instance_name": "shardon-test",
        "gpu_provider": "mock",
        "state_root": "state",
        "queue_poll_interval_seconds": 0.05,
        "backend_readiness_poll_interval_seconds": 0.1,
        "backend_stop_timeout_seconds": 5,
        "interactive_request_timeout_seconds": 30,
        "switch_grace_window_seconds": 0,
        **(global_overrides or {}),
    }
    _write(config / "router.yaml", global_config)
    _write(
        config / "gpu-inventory-enabled" / "gpu0.yaml",
        {"id": "gpu0", "uuid": "GPU-test-0", "pci_bus_id": "00000000:01:00.0"},
    )
    _write(
        config / "gpu-groups-enabled" / "rtx.yaml",
        {"id": "rtx", "display_name": "RTX", "gpu_ids": ["gpu0"], **(group or {})},
    )
    shutil.copy(
        source / "config" / "models-available" / "demo-chat-model.yaml",
        _ensure_dir(config / "models-enabled") / "demo-chat-model.yaml",
    )
    for backend in backends:
        _write(config / "backends-enabled" / f"{backend['id']}.yaml", backend)
    for item in deployments:
        _write(config / "deployments-enabled" / f"{item['id']}.yaml", item)
    (root / "state").mkdir()
    return root


def make_runtime(repo_root: Path, **provider_kwargs: Any) -> ShardonRuntime:
    return ShardonRuntime(repo_root=repo_root, gpu_provider=MockGPUProvider(**provider_kwargs))


def shutdown(runtime: ShardonRuntime) -> None:
    for deployment_id in list(runtime.backends.supervisor.processes):
        runtime.backends.supervisor.kill(deployment_id)
    runtime.stop_router()


def mock_stats(runtime: ShardonRuntime, deployment_id: str) -> dict[str, Any]:
    state = runtime.snapshot().deployments[deployment_id]
    adapter = runtime._deployment_adapter(
        runtime.config.deployments[deployment_id],
        state.selected_gpu_group_id or state.gpu_group_id,
    )
    return httpx.get(f"{adapter.backend.base_url}/mock/stats", timeout=5).json()


def events(runtime: ShardonRuntime, category: str) -> list[dict[str, Any]]:
    import json

    return [
        item
        for item in (json.loads(line) for line in runtime.read_events(tail_lines=10_000))
        if item["category"] == category
    ]


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write(path: Path, payload: dict[str, Any]) -> None:
    _ensure_dir(path.parent)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
