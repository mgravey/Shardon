from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from shardon_core.config.schemas import (
    AdminUserRecord,
    BackendRuntimeConfig,
    DeploymentConfig,
    GPUDeviceConfig,
    GPUGroupConfig,
    GlobalConfig,
    ModelConfig,
    RepositoryConfig,
)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return raw or {}


def _load_directory(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    items: list[dict[str, Any]] = []
    for file_path in sorted(path.glob("*.yaml")):
        payload = _load_yaml(file_path)
        if payload:
            items.append(payload)
    return items


def load_repository_config(config_root: Path) -> RepositoryConfig:
    global_config = GlobalConfig.model_validate(_load_yaml(config_root / "router.yaml"))
    backends = {
        item["id"]: BackendRuntimeConfig.model_validate(item)
        for item in _load_directory(config_root / "backends-enabled")
    }
    models = {
        item["id"]: ModelConfig.model_validate(item)
        for item in _load_directory(config_root / "models-enabled")
    }
    deployments = {
        item["id"]: DeploymentConfig.model_validate(item)
        for item in _load_directory(config_root / "deployments-enabled")
    }
    gpu_devices = {
        item["id"]: GPUDeviceConfig.model_validate(item)
        for item in _load_directory(config_root / "gpu-inventory-enabled")
    }
    gpu_groups = {
        item["id"]: GPUGroupConfig.model_validate(item)
        for item in _load_directory(config_root / "gpu-groups-enabled")
    }
    admin_users = {
        item["username"]: AdminUserRecord.model_validate(item)
        for item in _load_directory(config_root / "auth" / "admins-enabled")
    }
    policies = _load_yaml(config_root / "policies" / "scheduler.yaml")
    return RepositoryConfig(
        global_config=global_config,
        backends=backends,
        models=models,
        deployments=deployments,
        gpu_devices=gpu_devices,
        gpu_groups=gpu_groups,
        admin_users=admin_users,
        policies=policies,
    )
