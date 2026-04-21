from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Any

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from shardon_core.api.schemas import (
    APIKeySecretResponse,
    AdminLoginRequest,
    AdminLoginResponse,
    CreateAPIKeyRequest,
    DrainRequest,
    EnvironmentStatusResponse,
    ModelOnboardingRequest,
)
from shardon_core.config.schemas import BackendRuntimeConfig, DeploymentConfig, GPUDeviceConfig, GPUGroupConfig, ModelConfig
from shardon_core.services.container import build_container
from shardon_core.utils.env import load_dotenv_file


def _repo_root() -> Path:
    raw = os.environ.get("SHARDON_REPO_ROOT")
    if raw:
        return Path(raw).resolve()
    return Path(__file__).resolve().parents[4]


def create_app() -> FastAPI:
    load_dotenv_file(_repo_root() / ".env")
    app = FastAPI(title="Shardon Admin API", version="0.1.0")
    app.state.runtime = build_container(_repo_root())
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def get_runtime(request: Request):
        return request.app.state.runtime

    def admin_user(
        authorization: Annotated[str | None, Header()] = None,
        runtime=Depends(get_runtime),
    ) -> str:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="missing admin bearer token")
        username = runtime.admin_auth.validate_token(authorization.removeprefix("Bearer ").strip())
        if username is None:
            raise HTTPException(status_code=401, detail="invalid admin token")
        return username

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": "admin"}

    @app.post("/auth/login", response_model=AdminLoginResponse)
    async def login(payload: AdminLoginRequest, runtime=Depends(get_runtime)) -> AdminLoginResponse:
        token = runtime.admin_auth.authenticate(payload.username, payload.password)
        if token is None:
            raise HTTPException(status_code=401, detail="invalid credentials")
        return AdminLoginResponse(access_token=token)

    @app.get("/config/validate")
    async def validate_config(
        _: Annotated[str, Depends(admin_user)],
        runtime=Depends(get_runtime),
    ) -> dict[str, Any]:
        config = runtime.reload_config()
        return {
            "valid": True,
            "counts": {
                "backends": len(config.backends),
                "models": len(config.models),
                "deployments": len(config.deployments),
                "gpu_devices": len(config.gpu_devices),
                "gpu_groups": len(config.gpu_groups),
            },
        }

    @app.get("/resources")
    async def list_resources(
        _: Annotated[str, Depends(admin_user)],
        runtime=Depends(get_runtime),
    ) -> dict[str, Any]:
        config = runtime.config
        return {
            "backends": config.backends,
            "models": config.models,
            "deployments": config.deployments,
            "gpu_devices": config.gpu_devices,
            "gpu_groups": config.gpu_groups,
        }

    @app.put("/resources/backends/{backend_id}")
    async def put_backend(
        backend_id: str,
        payload: BackendRuntimeConfig,
        _: Annotated[str, Depends(admin_user)],
        runtime=Depends(get_runtime),
    ) -> dict[str, str]:
        runtime.upsert_config_item(collection="backends", item_id=backend_id, payload=payload.model_dump(mode="json"))
        return {"status": "ok"}

    @app.delete("/resources/backends/{backend_id}")
    async def delete_backend(
        backend_id: str,
        _: Annotated[str, Depends(admin_user)],
        runtime=Depends(get_runtime),
    ) -> dict[str, str]:
        runtime.delete_config_item(collection="backends", item_id=backend_id)
        return {"status": "ok"}

    @app.put("/resources/models/{model_id}")
    async def put_model(
        model_id: str,
        payload: ModelConfig,
        _: Annotated[str, Depends(admin_user)],
        runtime=Depends(get_runtime),
    ) -> dict[str, str]:
        runtime.upsert_config_item(collection="models", item_id=model_id, payload=payload.model_dump(mode="json"))
        return {"status": "ok"}

    @app.delete("/resources/models/{model_id}")
    async def delete_model(
        model_id: str,
        _: Annotated[str, Depends(admin_user)],
        runtime=Depends(get_runtime),
    ) -> dict[str, str]:
        runtime.delete_config_item(collection="models", item_id=model_id)
        return {"status": "ok"}

    @app.put("/resources/deployments/{deployment_id}")
    async def put_deployment(
        deployment_id: str,
        payload: DeploymentConfig,
        _: Annotated[str, Depends(admin_user)],
        runtime=Depends(get_runtime),
    ) -> dict[str, str]:
        runtime.upsert_config_item(
            collection="deployments",
            item_id=deployment_id,
            payload=payload.model_dump(mode="json"),
        )
        return {"status": "ok"}

    @app.delete("/resources/deployments/{deployment_id}")
    async def delete_deployment(
        deployment_id: str,
        _: Annotated[str, Depends(admin_user)],
        runtime=Depends(get_runtime),
    ) -> dict[str, str]:
        runtime.delete_config_item(collection="deployments", item_id=deployment_id)
        return {"status": "ok"}

    @app.put("/resources/gpu-groups/{group_id}")
    async def put_gpu_group(
        group_id: str,
        payload: GPUGroupConfig,
        _: Annotated[str, Depends(admin_user)],
        runtime=Depends(get_runtime),
    ) -> dict[str, str]:
        runtime.upsert_config_item(
            collection="gpu-groups",
            item_id=group_id,
            payload=payload.model_dump(mode="json"),
        )
        return {"status": "ok"}

    @app.delete("/resources/gpu-groups/{group_id}")
    async def delete_gpu_group(
        group_id: str,
        _: Annotated[str, Depends(admin_user)],
        runtime=Depends(get_runtime),
    ) -> dict[str, str]:
        runtime.delete_config_item(collection="gpu-groups", item_id=group_id)
        return {"status": "ok"}

    @app.put("/resources/gpu-devices/{gpu_id}")
    async def put_gpu_device(
        gpu_id: str,
        payload: GPUDeviceConfig,
        _: Annotated[str, Depends(admin_user)],
        runtime=Depends(get_runtime),
    ) -> dict[str, str]:
        runtime.upsert_config_item(
            collection="gpu-inventory",
            item_id=gpu_id,
            payload=payload.model_dump(mode="json"),
        )
        return {"status": "ok"}

    @app.delete("/resources/gpu-devices/{gpu_id}")
    async def delete_gpu_device(
        gpu_id: str,
        _: Annotated[str, Depends(admin_user)],
        runtime=Depends(get_runtime),
    ) -> dict[str, str]:
        runtime.delete_config_item(collection="gpu-inventory", item_id=gpu_id)
        return {"status": "ok"}

    @app.get("/api-keys")
    async def list_api_keys(
        _: Annotated[str, Depends(admin_user)],
        runtime=Depends(get_runtime),
    ) -> Any:
        return runtime.api_keys.list_keys()

    @app.post("/api-keys", response_model=APIKeySecretResponse)
    async def create_api_key(
        payload: CreateAPIKeyRequest,
        username: Annotated[str, Depends(admin_user)],
        runtime=Depends(get_runtime),
    ) -> APIKeySecretResponse:
        record, secret = runtime.api_keys.create_key(
            key_id=payload.key_id,
            user_name=payload.user_name,
            priority=payload.priority,
            permissions=payload.permissions,
            actor=username,
            metadata=payload.metadata,
        )
        return APIKeySecretResponse(id=record.id, secret=secret, prefix=record.secret_prefix)

    @app.delete("/api-keys/{key_id}")
    async def revoke_api_key(
        key_id: str,
        username: Annotated[str, Depends(admin_user)],
        runtime=Depends(get_runtime),
    ) -> dict[str, str]:
        record = runtime.api_keys.revoke_key(key_id, username)
        if record is None:
            raise HTTPException(status_code=404, detail="key not found")
        return {"status": "ok"}

    @app.get("/runtime/status")
    async def runtime_status(
        _: Annotated[str, Depends(admin_user)],
        runtime=Depends(get_runtime),
    ) -> dict[str, Any]:
        runtime.refresh_gpu_observations()
        runtime.enforce_keep_free()
        await runtime.refresh_backend_health()
        return runtime.status()

    @app.get("/runtime/environment", response_model=EnvironmentStatusResponse)
    async def runtime_environment(
        _: Annotated[str, Depends(admin_user)],
        runtime=Depends(get_runtime),
    ) -> EnvironmentStatusResponse:
        return EnvironmentStatusResponse.model_validate(runtime.environment_status())

    @app.get("/runtime/logs/{deployment_id}")
    async def runtime_logs(
        deployment_id: str,
        _: Annotated[str, Depends(admin_user)],
        runtime=Depends(get_runtime),
    ) -> dict[str, Any]:
        return {"deployment_id": deployment_id, "lines": runtime.read_backend_log(deployment_id)}

    @app.get("/runtime/events")
    async def runtime_events(
        _: Annotated[str, Depends(admin_user)],
        runtime=Depends(get_runtime),
    ) -> dict[str, Any]:
        return {"lines": runtime.read_events()}

    @app.post("/runtime/drain/{gpu_group_id}")
    async def drain_group(
        gpu_group_id: str,
        payload: DrainRequest,
        _: Annotated[str, Depends(admin_user)],
        runtime=Depends(get_runtime),
    ) -> Any:
        if gpu_group_id not in runtime.config.gpu_groups:
            raise HTTPException(status_code=404, detail="unknown gpu group")
        return await runtime.drain_group(gpu_group_id, payload.timeout_seconds)

    @app.post("/workflows/model-onboarding")
    async def onboard_model(
        payload: ModelOnboardingRequest,
        username: Annotated[str, Depends(admin_user)],
        runtime=Depends(get_runtime),
    ) -> dict[str, Any]:
        backend_compatibility = sorted(set(payload.backend_compatibility))
        model_payload = ModelConfig(
            id=payload.model_id,
            source=payload.source,
            display_name=payload.display_name,
            backend_compatibility=backend_compatibility,
            tasks=payload.tasks,
            tokenizer=payload.tokenizer,
            metadata=payload.metadata,
        ).model_dump(mode="json")

        deployment_payload: dict[str, Any] | None = None
        if payload.create_deployment:
            missing = [
                name
                for name, value in {
                    "deployment_id": payload.deployment_id,
                    "api_model_name": payload.api_model_name,
                    "deployment_display_name": payload.deployment_display_name,
                    "backend_runtime_id": payload.backend_runtime_id,
                    "gpu_group_id": payload.gpu_group_id,
                }.items()
                if not value
            ]
            if missing:
                raise HTTPException(
                    status_code=422,
                    detail={"error": "missing deployment fields", "fields": missing},
                )
            if payload.backend_runtime_id not in runtime.config.backends:
                raise HTTPException(status_code=404, detail="unknown backend runtime")
            if payload.gpu_group_id not in runtime.config.gpu_groups:
                raise HTTPException(status_code=404, detail="unknown gpu group")
            deployment_payload = DeploymentConfig(
                id=payload.deployment_id,
                model_id=payload.model_id,
                backend_runtime_id=payload.backend_runtime_id,
                gpu_group_id=payload.gpu_group_id,
                api_model_name=payload.api_model_name,
                display_name=payload.deployment_display_name,
                memory_fraction=payload.memory_fraction,
                enabled=payload.enabled,
                priority_weight=payload.priority_weight,
                tasks=payload.tasks,
            ).model_dump(mode="json")

        runtime.onboard_model(
            model_payload=model_payload,
            deployment_payload=deployment_payload,
            actor=username,
        )
        return {
            "status": "ok",
            "model_id": payload.model_id,
            "deployment_id": payload.deployment_id if payload.create_deployment else None,
        }

    @app.post("/debug/reload-config")
    async def reload_config(
        _: Annotated[str, Depends(admin_user)],
        runtime=Depends(get_runtime),
    ) -> dict[str, str]:
        runtime.reload_config()
        return {"status": "ok"}

    return app


def main() -> None:
    app = create_app()
    runtime = app.state.runtime
    uvicorn.run(
        app,
        host=runtime.config.global_config.admin_host,
        port=runtime.config.global_config.admin_port,
    )
