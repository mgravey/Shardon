# Changes

## 2026-11-04

- Fix tool calling through the router. `/v1/chat/completions` silently dropped every field not in its schema, including `tools`, `tool_choice` and `stream`, and rejected assistant `tool_calls` / tool `tool_call_id` messages. The chat schema now types `tools`, `tool_choice`, `parallel_tool_calls`, tool-call messages, the `developer` role, multimodal/`null` content, and forwards any other OpenAI/vLLM field (`top_p`, `stop`, `seed`, `chat_template_kwargs`, ...). Messages are forwarded exactly as sent, with no injected `null` fields. An invalid `tools`/`tool_choice` now gets a 422.
- Stream chat completions (`stream: true`) via SSE, so streamed `tool_calls` deltas and `finish_reason` arrive unchanged. Before, streamed chat requests got a single non-streamed JSON body.
- Return backend 4xx rejections with the backend's status code and body (`backend_response`), e.g. vLLM's "`auto` tool choice requires `--enable-auto-tool-choice`", instead of a generic 409.

## 2026-11-03

- Add a minimal Shardon SVG favicon: a mint S-shaped routing path with a bright active-model node, using the dashboard palette. Wire it into the admin UI with a base-aware URL for development and production builds.

## 2026-11-02

- Do not report a backend as `failed` when it was stopped by a router restart. Each loaded deployment records the router instance that launched its process (`process_owner_instance_id`). When reconciliation finds the process gone and that router has exited, the deployment becomes `unloaded` with reason `backend stopped with its router (router restart)` and a `backend.recovery` event with `recovery_result: unloaded_with_router`. A process that dies while its router is alive, or that has no recorded owner, is still `failed`. A router adopts backends that outlived their launching router.
- Show why a deployment is failed or unloaded: the activity API returns `state_reason`, shown as the status chip tooltip on the activity board; the admin deployments table shows the last transition reason on hover.

## 2026-11-01

- Add opt-in vLLM sleep/wake orchestration (`sleep` on backends, overridable per deployment): idle sleep after `idle_seconds` (default 10 minutes), wake on demand shared by concurrent requests, level 1 and level 2 wake sequences confirmed by `/is_sleeping`, wake time counted against the interactive timeout, and a cold-restart fallback when a wake fails.
- Add `falling_asleep`, `sleeping` and `waking` deployment states to status APIs, events, the activity view and the admin UI. Sleeping deployments stay loaded, with resident memory reduced to the sleeping residual.
- Plan wakes like loads, and evict sleeping deployments before awake ones. `sleep_on_evict` sleeps instead of stopping when the incoming deployment uses another port.
- Bound sleepers with `gpu_groups.<id>.max_sleeping`, `max_sleeping_deployments` and `sleep.max_sleep_seconds`; the least recently used sleepers are stopped completely.
- Add admin sleep/wake endpoints, per-deployment Sleep/Wake/Stop/Load buttons, and a public no-admin Sleep button on the activity board. The button is shown only while a deployment is awake with nothing in flight, queued or batched; Stop stays admin-only. The router never proxies vLLM dev endpoints.
- Add the vLLM sleep API to the `mock-vllm-v2` demo runtime (`--enable-sleep-mode`), with fault injection, plus process-level tests.
- Add per-deployment ports (`per_deployment_ports` starting at the `base_url` port and adding 1 per deployment, or `port_range` on backends; `port` on deployments). Ports are allocated under the state lock, passed as `IGF_SHARDON_RUNTIME_PORT`, and released on stop.
- Add `memory_model: elastic` GPU groups for kvcached co-residency. Admission uses resident footprints, learned from per-process `nvidia-smi` memory plus the kvcached reserve, together with observed free memory. Nothing is evicted when the new deployment fits; otherwise idle deployments are slept, then the least recently used are stopped. `static` groups are unchanged.
- Add `kvcached` controls (`limit`, `limit_percent`, `min`, `priority`, `headroom`) applied with a configurable `kvctl`, a default `KVCACHED_IPC_NAME` of `igf-shardon-<deployment id>`, and live KV usage in the status APIs and the UI.

## 2026-10-14

- Track router instance ownership with process-held locks and recover interrupted requests on router startup and scheduler ticks, preserving live and unverifiable owners.
- Remove recovered queue, active-request, and deployment references together; retain interruption reasons and deduplicate usage across recovery retries, including partially committed batch progress.
- Close prefetched streaming iterators on client disconnects and failed sends, including failures before headers.
- Add real SIGKILL/restart, live-owner, interrupted-accounting, batch recovery, and streaming disconnect regression tests.

## 2026-10-10

- Forward optional chat response-format and reasoning-effort options to backends.
- Avoid restarting backends on read, write, and pool timeouts; retain recovery for connection failures and include exception types in request diagnostics.
- Make non-streaming JSON inference read/connect timeouts configurable per backend and GPU group, defaulting to 600/10 seconds.
- Remove completed/failed requests from the queue in the same state mutation as finalization.
- Raise FastAPI, Pydantic, Uvicorn, and PyYAML minimums for Python 3.14 while retaining Python 3.11+ support.

## 2026-08-16

- Fixed dynamic backend recovery when a tracked process exits or a still-running parent loses its serving child and the backend health endpoint becomes unreachable.
- Added single-flight per-deployment lifecycle locking so concurrent requests share one backend start/recovery operation.
- Added a strict one-retry policy for connection-level failures in non-streaming requests and in streaming requests before the first emitted chunk; HTTP status errors do not trigger recovery.
- Made the router background scheduler survive and report failed health/scheduling ticks instead of terminating its task.
- Added lifecycle recovery diagnostics and mocked regression coverage for dead PIDs, unreachable health, bounded request retry, concurrent starts, streaming recovery, HTTP `4xx` behavior, manual unload/on-demand reload, and background-loop resilience.

## 2026-04-27

- Added router support for `POST /v1/responses` and forwards Responses API create payloads to OpenAI-compatible backends.
- Added streaming passthrough for `POST /v1/responses` requests with `stream: true`, returning backend SSE bytes as `text/event-stream`.
- Added streaming passthrough for `POST /v1/completions` requests with `stream: true`, returning backend SSE bytes as `text/event-stream`.
- Added `scripts/smoke_openai_surfaces.py` for live router checks covering switching, Completions streaming, Responses API, and Responses streaming.
- Added the `response` scheduler task and backend capability flag so deployments can explicitly opt into the Responses API surface.
- Updated demo chat model/deployments and mock vLLM backends to advertise Responses API support.

## 2026-04-22

- Added multimodal routing scaffolding across scheduler, runtime, backend adapters, and status output.
- Added OpenAI-style audio endpoints on router API:
  - `POST /v1/audio/speech`
  - `POST /v1/audio/transcriptions`
  - `POST /v1/audio/translations`
- Added backend adapter support for multipart audio upload flows and binary speech responses.
- Added WhisperX adapter translation layer for OpenAI-compatible transcription/translation outputs.
- Added first-class modality metadata on models/deployments/backends (`text`, `audio`, `image`, `video`) and capability-aware candidate filtering.
- Added regression coverage for dependency binding and audio JSON/multipart routes, plus scheduler/config capability behavior.
- Added `python-multipart` to `shardon-router-api` package dependencies so multipart audio routes start cleanly after standard bootstrap.
- Added regression checks to ensure router multipart dependency is declared and bootstrap keeps syncing all workspace packages.
- Interactive routing now rejects `no compatible deployment` requests immediately (HTTP 404) and does not queue unsupported/unconfigured models.
- Added per-model runtime launch flags (`runtime_launch_args` and `runtime_launch_args_by_backend_type`) so model-specific backend CLI flags can be appended automatically at startup.
- Fixed process supervision stop behavior so `backend_stop_timeout_seconds` is treated as a true max timeout and does not force waiting when the backend exits quickly.
