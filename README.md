# Shardon

`Shardon — pardon, one model at a time.`

Shardon is a Linux-first self-hosted LLM router and admin platform built for constrained GPU environments. It exposes an OpenAI-compatible inference API, keeps admin/control concerns on a separate service, and dynamically loads, unloads, and schedules model deployments across GPU groups and backend runtimes.

## What is in this MVP

- Separate FastAPI services for admin and router planes.
- File-backed desired state in YAML and runtime state in JSON/JSONL.
- Scheduler with memory-aware deployment admission, LRU eviction, drain handling, keep-free enforcement, and a model-switch grace window.
- Backend abstraction for vLLM, SGLang, and independently runnable runtime folders.
- OpenAI-compatible endpoints for `models`, `chat/completions`, `completions`, `embeddings`, and `batches`.
- Admin UI for configuration, status, drains, keys, requests, jobs, and events.
- Demo mock runtimes for local development without GPUs.

## Monorepo

- [apps/admin_api](/Users/mathieugravey/projects/Shardon/apps/admin_api)
- [apps/router_api](/Users/mathieugravey/projects/Shardon/apps/router_api)
- [apps/admin_web](/Users/mathieugravey/projects/Shardon/apps/admin_web)
- [packages/shardon_core](/Users/mathieugravey/projects/Shardon/packages/shardon_core)
- [config](/Users/mathieugravey/projects/Shardon/config)
- [state](/Users/mathieugravey/projects/Shardon/state)
- [demo](/Users/mathieugravey/projects/Shardon/demo)
- [docs](/Users/mathieugravey/projects/Shardon/docs)

## Quick Start

1. Install Python dependencies with `uv sync`.
2. Install frontend dependencies with `npm install`.
3. Optional: create a repo `.env` from [.env.example](/Users/mathieugravey/projects/Shardon/.env.example) and set `HF_TOKEN` if backends need to download models from Hugging Face.
4. Start the admin API with `uv run --package shardon-admin-api shardon-admin-api`.
5. Start the router API with `uv run --package shardon-router-api shardon-router-api`.
6. Start the admin web UI with `npm --workspace apps/admin_web run dev`.

Default ports:

- Admin API: `http://127.0.0.1:8081`
- Router API: `http://127.0.0.1:8080`
- Admin UI: `http://127.0.0.1:5173`

## Core Ideas

- YAML is desired state.
- JSON and JSONL are observed state.
- Backends live in runtime folders and can be run with or without Shardon.
- GPU groups are first-class scheduling targets.
- `keep_free` is enforced aggressively from observed process ownership.
- Drains are blocking runtime operations, not long-lived reservations.

## Model Downloads

- `HF_TOKEN` is read from the process environment or from a repo-local `.env` file.
- The token is not stored in YAML, runtime JSON, or the admin UI.
- Backend subprocesses inherit `HF_TOKEN`, `HF_HOME`, and the model source information when launched.
- The admin UI now includes a guided model onboarding form and shows only whether `HF_TOKEN` is configured, never the secret itself.

## Documentation

- [Architecture](/Users/mathieugravey/projects/Shardon/docs/architecture.md)
- [Configuration](/Users/mathieugravey/projects/Shardon/docs/configuration.md)
- [Scheduling](/Users/mathieugravey/projects/Shardon/docs/scheduling.md)
- [GPU Mapping](/Users/mathieugravey/projects/Shardon/docs/gpu-mapping.md)
- [Keep-Free](/Users/mathieugravey/projects/Shardon/docs/keep-free.md)
- [Drain](/Users/mathieugravey/projects/Shardon/docs/drain.md)
- [Batching](/Users/mathieugravey/projects/Shardon/docs/batch.md)
- [Debugging](/Users/mathieugravey/projects/Shardon/docs/debugging.md)
- [Running Backends Outside Shardon](/Users/mathieugravey/projects/Shardon/docs/backends-outside-shardon.md)
