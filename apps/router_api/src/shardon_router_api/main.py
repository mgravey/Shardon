from __future__ import annotations

import asyncio
import contextlib
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Request

from shardon_core.api.schemas import BatchCreateRequest, ChatCompletionRequest, CompletionRequest, EmbeddingRequest
from shardon_core.auth.service import AuthResult
from shardon_core.services.container import build_container
from shardon_core.services.runtime import RuntimeOperationError, ShardonRuntime
from shardon_core.utils.env import load_dotenv_file


def _repo_root() -> Path:
    raw = os.environ.get("SHARDON_REPO_ROOT")
    if raw:
        return Path(raw).resolve()
    return Path(__file__).resolve().parents[4]


@asynccontextmanager
async def lifespan(app: FastAPI):
    stop_event = asyncio.Event()
    runtime = app.state.runtime

    async def background() -> None:
        while not stop_event.is_set():
            runtime.refresh_gpu_observations()
            runtime.enforce_keep_free()
            await runtime.refresh_backend_health()
            await runtime.process_one_batch()
            await asyncio.sleep(runtime.config.global_config.scheduler_tick_seconds)

    task = asyncio.create_task(background())
    try:
        yield
    finally:
        stop_event.set()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


def create_app() -> FastAPI:
    load_dotenv_file(_repo_root() / ".env")
    app = FastAPI(title="Shardon Router API", version="0.1.0", lifespan=lifespan)
    app.state.runtime = build_container(_repo_root())

    def get_runtime(request: Request) -> ShardonRuntime:
        return request.app.state.runtime

    def api_key_auth(
        authorization: Annotated[str | None, Header()] = None,
        runtime: ShardonRuntime = Depends(get_runtime),
    ) -> AuthResult:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail={"error": "missing api key"})
        auth = runtime.api_keys.authenticate(authorization.removeprefix("Bearer ").strip())
        if auth is None:
            raise HTTPException(status_code=401, detail={"error": "invalid api key"})
        return auth

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": "router"}

    @app.get("/v1/models")
    async def list_models(
        auth: AuthResult = Depends(api_key_auth),
        runtime: ShardonRuntime = Depends(get_runtime),
    ) -> dict[str, Any]:
        _ = auth
        return {"object": "list", "data": runtime.list_api_models()}

    @app.post("/v1/chat/completions")
    async def chat_completions(
        payload: ChatCompletionRequest,
        auth: AuthResult = Depends(api_key_auth),
        runtime: ShardonRuntime = Depends(get_runtime),
    ) -> Any:
        try:
            return await runtime.route_chat(payload, auth)
        except RuntimeOperationError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    @app.post("/v1/completions")
    async def completions(
        payload: CompletionRequest,
        auth: AuthResult = Depends(api_key_auth),
        runtime: ShardonRuntime = Depends(get_runtime),
    ) -> Any:
        try:
            return await runtime.route_completion(payload, auth)
        except RuntimeOperationError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    @app.post("/v1/embeddings")
    async def embeddings(
        payload: EmbeddingRequest,
        auth: AuthResult = Depends(api_key_auth),
        runtime: ShardonRuntime = Depends(get_runtime),
    ) -> Any:
        try:
            return await runtime.route_embedding(payload, auth)
        except RuntimeOperationError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    @app.post("/v1/batches")
    async def create_batch(
        payload: BatchCreateRequest,
        auth: AuthResult = Depends(api_key_auth),
        runtime: ShardonRuntime = Depends(get_runtime),
    ) -> Any:
        job = await runtime.submit_batch(payload, auth)
        return {"id": job.id, "object": "batch", "status": job.status}

    @app.get("/v1/batches/{batch_id}")
    async def get_batch(
        batch_id: str,
        auth: AuthResult = Depends(api_key_auth),
        runtime: ShardonRuntime = Depends(get_runtime),
    ) -> Any:
        snapshot = runtime.snapshot()
        job = snapshot.batch_jobs.get(batch_id)
        if job is None:
            raise HTTPException(status_code=404, detail={"error": "batch not found"})
        if job.api_key_id != auth.id:
            raise HTTPException(status_code=403, detail={"error": "forbidden"})
        return job

    @app.get("/shardon/status")
    async def shardon_status(
        auth: AuthResult = Depends(api_key_auth),
        runtime: ShardonRuntime = Depends(get_runtime),
    ) -> Any:
        _ = auth
        runtime.refresh_gpu_observations()
        runtime.enforce_keep_free()
        await runtime.refresh_backend_health()
        return runtime.status()

    @app.get("/shardon/batches/{batch_id}/progress")
    async def batch_progress(
        batch_id: str,
        auth: AuthResult = Depends(api_key_auth),
        runtime: ShardonRuntime = Depends(get_runtime),
    ) -> Any:
        snapshot = runtime.snapshot()
        job = snapshot.batch_jobs.get(batch_id)
        if job is None:
            raise HTTPException(status_code=404, detail={"error": "batch not found"})
        if job.api_key_id != auth.id:
            raise HTTPException(status_code=403, detail={"error": "forbidden"})
        return {
            "id": job.id,
            "status": job.status,
            "total_items": job.total_items,
            "completed_items": job.completed_items,
            "failed_items": job.failed_items,
            "deployment_id": job.deployment_id,
            "updated_at": job.updated_at,
        }

    return app


def main() -> None:
    app = create_app()
    runtime = app.state.runtime
    uvicorn.run(
        app,
        host=runtime.config.global_config.router_host,
        port=runtime.config.global_config.router_port,
    )
