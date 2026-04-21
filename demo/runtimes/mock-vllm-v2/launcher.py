from __future__ import annotations

import argparse
import os
import socket
import time

import uvicorn
from fastapi import FastAPI


def create_app(runtime_label: str) -> FastAPI:
    app = FastAPI(title=runtime_label)

    @app.get("/health")
    async def health() -> dict[str, object]:
        return {
            "status": "ok",
            "runtime_label": runtime_label,
            "hostname": socket.gethostname(),
            "deployment_id": os.environ.get("SHARDON_DEPLOYMENT_ID"),
            "model_id": os.environ.get("SHARDON_MODEL_ID"),
        }

    @app.post("/v1/chat/completions")
    async def chat(payload: dict[str, object]) -> dict[str, object]:
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

    @app.post("/v1/completions")
    async def completion(payload: dict[str, object]) -> dict[str, object]:
        prompt = payload.get("prompt", "")
        return {
            "id": f"cmpl-{int(time.time())}",
            "object": "text_completion",
            "model": payload.get("model", os.environ.get("SHARDON_API_MODEL_NAME", "demo-chat")),
            "choices": [{"index": 0, "text": f"[{runtime_label}] fast completion for {prompt}", "finish_reason": "stop"}],
        }

    @app.post("/v1/embeddings")
    async def embeddings(payload: dict[str, object]) -> dict[str, object]:
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
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--runtime-label", required=True)
    args = parser.parse_args()
    uvicorn.run(create_app(args.runtime_label), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()

