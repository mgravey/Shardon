from __future__ import annotations

import argparse
import os
import socket
import time

import asyncio

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def create_app(runtime_label: str, *, enable_sleep_mode: bool = False) -> FastAPI:
    app = FastAPI(title=runtime_label)
    # Mock of vLLM sleep mode (--enable-sleep-mode + VLLM_SERVER_DEV_MODE=1).
    # Fault injection for tests: MOCK_WAKE_FAIL, MOCK_WAKE_DELAY_SECONDS,
    # MOCK_SLEEP_DELAY_SECONDS.
    sleep_state: dict[str, object] = {
        "sleeping": False,
        "level": None,
        "weights_allocated": True,
        "weights_loaded": True,
        "kv_cache_allocated": True,
    }
    stats: dict[str, int] = {
        "sleep_calls": 0,
        "wake_calls": 0,
        "reload_calls": 0,
        "is_sleeping_calls": 0,
        "inference_calls": 0,
        "inference_while_sleeping": 0,
        "health_calls": 0,
    }
    app.state.sleep_state = sleep_state
    app.state.stats = stats

    def guard_inference() -> None:
        stats["inference_calls"] += 1
        if sleep_state["sleeping"]:
            stats["inference_while_sleeping"] += 1
            raise HTTPException(status_code=503, detail="engine is sleeping")
        if not sleep_state["weights_loaded"]:
            raise HTTPException(status_code=500, detail="weights were not reloaded after sleep")

    @app.get("/mock/stats")
    async def mock_stats() -> dict[str, object]:
        return {**stats, **sleep_state, "pid": os.getpid()}

    if enable_sleep_mode:

        @app.post("/sleep")
        async def sleep(level: int = 1, mode: str = "wait") -> dict[str, object]:
            _ = mode
            stats["sleep_calls"] += 1
            await asyncio.sleep(float(os.environ.get("MOCK_SLEEP_DELAY_SECONDS", "0")))
            sleep_state.update(
                sleeping=True,
                level=level,
                kv_cache_allocated=False,
                weights_allocated=False,
                # Level 1 offloads weights to CPU; level 2 discards them.
                weights_loaded=level == 1,
            )
            return {}

        @app.post("/wake_up")
        async def wake_up(tags: list[str] | None = Query(default=None)) -> dict[str, object]:
            stats["wake_calls"] += 1
            await asyncio.sleep(float(os.environ.get("MOCK_WAKE_DELAY_SECONDS", "0")))
            if _env_flag("MOCK_WAKE_FAIL"):
                raise HTTPException(status_code=500, detail="injected wake failure")
            selected = set(tags or ["weights", "kv_cache"])
            if "weights" in selected:
                sleep_state["weights_allocated"] = True
            if "kv_cache" in selected:
                sleep_state["kv_cache_allocated"] = True
            if sleep_state["weights_allocated"] and sleep_state["kv_cache_allocated"]:
                sleep_state["sleeping"] = False
            return {}

        @app.post("/collective_rpc")
        async def collective_rpc(request: Request) -> dict[str, object]:
            payload = await request.json()
            if payload.get("method") == "reload_weights":
                stats["reload_calls"] += 1
                if not sleep_state["weights_allocated"]:
                    raise HTTPException(status_code=400, detail="weights not allocated")
                sleep_state["weights_loaded"] = True
            return {"results": [None]}

        @app.get("/is_sleeping")
        async def is_sleeping() -> dict[str, bool]:
            stats["is_sleeping_calls"] += 1
            return {"is_sleeping": bool(sleep_state["sleeping"])}

    @app.get("/v1/models")
    async def models() -> dict[str, object]:
        # Like vLLM, listing models works while asleep and never wakes it.
        stats["health_calls"] += 1
        name = os.environ.get("SHARDON_API_MODEL_NAME", "demo-chat")
        return {"object": "list", "data": [{"id": name, "object": "model"}]}

    @app.get("/health")
    async def health() -> dict[str, object]:
        stats["health_calls"] += 1
        return {
            "status": "ok",
            "runtime_label": runtime_label,
            "hostname": socket.gethostname(),
            "deployment_id": os.environ.get("SHARDON_DEPLOYMENT_ID"),
            "model_id": os.environ.get("SHARDON_MODEL_ID"),
        }

    @app.post("/v1/chat/completions")
    async def chat(payload: dict[str, object]) -> dict[str, object]:
        guard_inference()
        prompt = " ".join(str(item.get("content", "")) for item in payload.get("messages", []))
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "model": payload.get("model", os.environ.get("SHARDON_API_MODEL_NAME", "demo-chat")),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": f"[{runtime_label}] newer vLLM mock reply: {prompt[:120]}",
                    },
                    "finish_reason": "stop",
                }
            ],
        }

    @app.post("/v1/responses")
    async def response(payload: dict[str, object]) -> dict[str, object]:
        guard_inference()
        prompt = str(payload.get("input", ""))
        return {
            "id": f"resp-{int(time.time())}",
            "object": "response",
            "model": payload.get("model", os.environ.get("SHARDON_API_MODEL_NAME", "demo-chat")),
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": f"[{runtime_label}] Shardon demo response: {prompt[:120]}",
                        }
                    ],
                }
            ],
        }

    @app.post("/v1/completions")
    async def completion(payload: dict[str, object]) -> dict[str, object]:
        guard_inference()
        prompt = payload.get("prompt", "")
        return {
            "id": f"cmpl-{int(time.time())}",
            "object": "text_completion",
            "model": payload.get("model", os.environ.get("SHARDON_API_MODEL_NAME", "demo-chat")),
            "choices": [{"index": 0, "text": f"[{runtime_label}] fast completion for {prompt}", "finish_reason": "stop"}],
        }

    @app.post("/v1/embeddings")
    async def embeddings(payload: dict[str, object]) -> dict[str, object]:
        guard_inference()
        raw = payload.get("input", "")
        items = raw if isinstance(raw, list) else [raw]
        return {
            "object": "list",
            "data": [
                {"object": "embedding", "index": index, "embedding": [float(index), 1.1, 1.2, 1.3]}
                for index, _ in enumerate(items)
            ],
            "model": payload.get("model", os.environ.get("SHARDON_API_MODEL_NAME", "demo-embed")),
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--runtime-label", required=True)
    parser.add_argument("--enable-sleep-mode", action="store_true")
    args = parser.parse_args()
    port = args.port
    if port is None:
        # Shardon passes the allocated per-deployment port when the backend
        # declares a port range (see docs/sleep-and-elastic-memory.md).
        raw = os.environ.get("IGF_SHARDON_RUNTIME_PORT")
        if not raw:
            parser.error("--port is required unless IGF_SHARDON_RUNTIME_PORT is set")
        port = int(raw)
    uvicorn.run(
        create_app(args.runtime_label, enable_sleep_mode=args.enable_sleep_mode),
        host="127.0.0.1",
        port=port,
    )


if __name__ == "__main__":
    main()
