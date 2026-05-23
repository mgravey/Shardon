# Changes

## 2026-04-27

- Added router support for `POST /v1/responses` and forwards Responses API create payloads to OpenAI-compatible backends.
- Added streaming passthrough for `POST /v1/responses` requests with `stream: true`, returning backend SSE bytes as `text/event-stream`.
- Added streaming passthrough for `POST /v1/completions` requests with `stream: true`, returning backend SSE bytes as `text/event-stream`.
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
