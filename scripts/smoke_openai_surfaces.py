#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

try:
    import httpx
except ImportError as exc:  # pragma: no cover - operator setup guard
    raise SystemExit(
        "Missing dependency: httpx. Run this with the project environment, for example "
        "`.venv/bin/python scripts/smoke_openai_surfaces.py`."
    ) from exc


@dataclass(frozen=True)
class SmokeConfig:
    router_url: str
    api_key: str
    primary_model: str
    switch_model: str | None
    timeout_seconds: float
    max_stream_events: int
    require_done: bool


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke-test Shardon router switching and OpenAI-compatible completions/"
            "responses streaming passthrough."
        )
    )
    parser.add_argument(
        "--router-url",
        default=os.environ.get("SHARDON_ROUTER_URL", "http://127.0.0.1:8080"),
        help="Router base URL. Defaults to SHARDON_ROUTER_URL or http://127.0.0.1:8080.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("SHARDON_API_KEY"),
        help="Router API key. Defaults to SHARDON_API_KEY.",
    )
    parser.add_argument(
        "--model",
        dest="primary_model",
        default=os.environ.get("SHARDON_SMOKE_MODEL", "demo-chat"),
        help="Primary model name to request. Defaults to SHARDON_SMOKE_MODEL or demo-chat.",
    )
    parser.add_argument(
        "--switch-model",
        default=os.environ.get("SHARDON_SMOKE_SWITCH_MODEL"),
        help=(
            "Optional second model name used for the switching check. If omitted, the "
            "script verifies repeat routing on the primary model."
        ),
    )
    parser.add_argument(
        "--timeout",
        dest="timeout_seconds",
        type=float,
        default=float(os.environ.get("SHARDON_SMOKE_TIMEOUT", "120")),
        help="Per-request timeout in seconds. Defaults to SHARDON_SMOKE_TIMEOUT or 120.",
    )
    parser.add_argument(
        "--max-stream-events",
        type=int,
        default=int(os.environ.get("SHARDON_SMOKE_MAX_STREAM_EVENTS", "8")),
        help="Maximum SSE data events to read from each streaming endpoint.",
    )
    parser.add_argument(
        "--require-done",
        action="store_true",
        default=os.environ.get("SHARDON_SMOKE_REQUIRE_DONE", "").lower() in {"1", "true", "yes"},
        help="Require a final `data: [DONE]` event in streaming responses.",
    )
    return parser


def _headers(config: SmokeConfig) -> dict[str, str]:
    return {"Authorization": f"Bearer {config.api_key}"}


def _compact_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _assert_status(response: httpx.Response, expected: int = 200) -> None:
    if response.status_code == expected:
        return
    content_type = response.headers.get("content-type", "")
    body = response.text
    if "application/json" in content_type:
        with contextlib.suppress(ValueError):
            body = _compact_json(response.json())
    raise AssertionError(
        f"{response.request.method} {response.request.url} returned "
        f"{response.status_code}, expected {expected}: {body}"
    )


def _response_text(payload: dict[str, Any]) -> str:
    output = payload.get("output")
    if isinstance(output, list):
        parts: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if isinstance(content, list):
                for content_item in content:
                    if isinstance(content_item, dict) and isinstance(content_item.get("text"), str):
                        parts.append(content_item["text"])
        return "".join(parts)
    return str(payload)


def _completion_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return str(payload)
    first = choices[0]
    if not isinstance(first, dict):
        return str(payload)
    text = first.get("text")
    return text if isinstance(text, str) else str(payload)


def _loaded_deployments(status: dict[str, Any]) -> list[str]:
    deployments = status.get("deployments", {})
    if not isinstance(deployments, dict):
        return []
    loaded = [
        deployment_id
        for deployment_id, state in deployments.items()
        if isinstance(state, dict) and state.get("loaded") is True
    ]
    return sorted(loaded)


def _print_step(name: str, detail: str = "") -> None:
    suffix = f" {detail}" if detail else ""
    print(f"[smoke] {name}{suffix}", flush=True)


def get_status(client: httpx.Client, config: SmokeConfig) -> dict[str, Any] | None:
    response = client.get("/shardon/status", headers=_headers(config))
    if response.status_code == 404:
        return None
    _assert_status(response)
    payload = response.json()
    return payload if isinstance(payload, dict) else None


def post_json(
    client: httpx.Client,
    config: SmokeConfig,
    path: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    response = client.post(path, headers=_headers(config), json=payload)
    _assert_status(response)
    result = response.json()
    if not isinstance(result, dict):
        raise AssertionError(f"{path} returned non-object JSON: {result!r}")
    return result


def get_json(client: httpx.Client, config: SmokeConfig, path: str) -> dict[str, Any]:
    response = client.get(path, headers=_headers(config))
    _assert_status(response)
    result = response.json()
    if not isinstance(result, dict):
        raise AssertionError(f"{path} returned non-object JSON: {result!r}")
    return result


def read_sse_stream(
    client: httpx.Client,
    config: SmokeConfig,
    path: str,
    payload: dict[str, Any],
) -> list[str]:
    events: list[str] = []
    with client.stream("POST", path, headers=_headers(config), json=payload) as response:
        _assert_status(response)
        content_type = response.headers.get("content-type", "")
        if "text/event-stream" not in content_type:
            raise AssertionError(f"{path} returned {content_type!r}, expected text/event-stream")
        for line in response.iter_lines():
            if not line or line.startswith(":"):
                continue
            if not line.startswith("data:"):
                continue
            data = line.removeprefix("data:").strip()
            events.append(data)
            if data == "[DONE]" or len(events) >= config.max_stream_events:
                break
    if not events:
        raise AssertionError(f"{path} did not yield any SSE data events")
    if config.require_done and events[-1] != "[DONE]":
        raise AssertionError(f"{path} did not finish with data: [DONE]; saw {events[-1]!r}")
    return events


def summarize_events(events: Iterable[str]) -> str:
    materialized = list(events)
    preview = materialized[:2]
    suffix = "..." if len(materialized) > len(preview) else ""
    return f"{len(materialized)} events {preview!r}{suffix}"


def run(config: SmokeConfig) -> None:
    timeout = httpx.Timeout(config.timeout_seconds)
    with httpx.Client(base_url=config.router_url.rstrip("/"), timeout=timeout) as client:
        _print_step("health")
        _assert_status(client.get("/health"))

        _print_step("models")
        models = get_json(client, config, "/v1/models")
        listed_models = [
            item.get("id") for item in models.get("data", []) if isinstance(item, dict)
        ]
        if config.primary_model not in listed_models:
            raise AssertionError(
                f"model {config.primary_model!r} not listed by /v1/models: {listed_models!r}"
            )

        before = get_status(client, config)
        before_loaded = _loaded_deployments(before) if before else []

        _print_step("switching", f"primary={config.primary_model}")
        first_completion = post_json(
            client,
            config,
            "/v1/completions",
            {"model": config.primary_model, "prompt": "Shardon switching smoke test"},
        )
        print(f"[smoke] completion text: {_completion_text(first_completion)[:160]}", flush=True)

        target_model = config.switch_model or config.primary_model
        if config.switch_model is None:
            _print_step("switching", "no --switch-model provided; verifying repeat routing")
        else:
            _print_step("switching", f"secondary={config.switch_model}")
        post_json(
            client,
            config,
            "/v1/completions",
            {"model": target_model, "prompt": "Shardon secondary routing smoke test"},
        )

        after = get_status(client, config)
        after_loaded = _loaded_deployments(after) if after else []
        if before is not None and after is not None:
            print(
                f"[smoke] loaded deployments: before={before_loaded!r} after={after_loaded!r}",
                flush=True,
            )

        _print_step("completion streaming")
        completion_events = read_sse_stream(
            client,
            config,
            "/v1/completions",
            {
                "model": config.primary_model,
                "prompt": "Stream a short completion through Shardon",
                "stream": True,
            },
        )
        print(f"[smoke] completion stream: {summarize_events(completion_events)}", flush=True)

        _print_step("responses api")
        response = post_json(
            client,
            config,
            "/v1/responses",
            {"model": config.primary_model, "input": "Return a short Responses API answer"},
        )
        print(f"[smoke] response text: {_response_text(response)[:160]}", flush=True)

        _print_step("responses streaming")
        response_events = read_sse_stream(
            client,
            config,
            "/v1/responses",
            {
                "model": config.primary_model,
                "input": "Stream a short Responses API answer through Shardon",
                "stream": True,
            },
        )
        print(f"[smoke] response stream: {summarize_events(response_events)}", flush=True)

    _print_step("ok")


def main() -> int:
    args = _parser().parse_args()
    if not args.api_key:
        print("error: --api-key or SHARDON_API_KEY is required", file=sys.stderr)
        return 2
    config = SmokeConfig(
        router_url=args.router_url,
        api_key=args.api_key,
        primary_model=args.primary_model,
        switch_model=args.switch_model,
        timeout_seconds=args.timeout_seconds,
        max_stream_events=args.max_stream_events,
        require_done=args.require_done,
    )
    try:
        run(config)
    except Exception as exc:
        print(f"[smoke] failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
