# Shardon

`Shardon — pardon, one model at a time.`

Shardon is a Linux-first self-hosted LLM router and admin platform built for constrained GPU environments. It exposes an OpenAI-compatible inference API, keeps admin/control concerns on a separate service, and dynamically loads, unloads, and schedules model deployments across GPU groups and backend runtimes.

## What is in this MVP

- Separate FastAPI services for admin and router planes.
- File-backed desired state in YAML and runtime state in JSON/JSONL.
- Scheduler with memory-aware deployment admission, LRU eviction, drain handling, keep-free enforcement, and a model-switch grace window.
- Backend abstraction for vLLM, SGLang, and independently runnable runtime folders.
- Deployments can declare an ordered list of eligible GPU groups, with runtime GPU-group selection at load/start time.
- Models, deployments, and backends can declare modality capabilities (`text`, `audio`, `image`, `video`) surfaced in `/v1/models` and runtime status.
- Models can declare per-model runtime launch flags (global and backend-type scoped) that are appended when Shardon starts the backend process.
- OpenAI-compatible endpoints for `models`, `responses`, `chat/completions`, `completions`, `embeddings`, `audio/speech`, `audio/transcriptions`, `audio/translations`, and `batches`.
- Streaming `POST /v1/responses` and `POST /v1/completions` requests (`stream: true`) are proxied as `text/event-stream` when the selected backend supports streaming.
- Admin UI for configuration, status, drains, keys, requests, jobs, and events.
- Demo mock runtimes for local development without GPUs.
- Route-level FastAPI dependencies declared directly with `= Depends(...)` for compatibility across FastAPI/Pydantic versions.

## Monorepo

- [apps/admin_api](apps/admin_api)
- [apps/router_api](apps/router_api)
- [apps/admin_web](apps/admin_web)
- [packages/shardon_core](packages/shardon_core)
- [config](config)
- [state](state)
- [demo](demo)
- [docs](docs)

## Install and First Run

Run the commands below from the repository root. The shipped configuration uses
mock backends, placeholder GPUs, and explicit `gpu_provider: mock`: it is suitable for trying the UI and routing
without downloading a model. For actual NVIDIA inference, complete the
[real GPU and backend setup](docs/installation.md#real-gpu-and-backend-setup),
including `gpu_provider: nvidia-smi` in deployment `config/router.yaml`.

### Prerequisites

| Component | Requirement |
| --- | --- |
| OS | Linux for GPU operation. macOS can run the mock demo and development tools; native Windows is not supported by the current POSIX process/file-lock code. |
| Python | 3.11+ for the Shardon services; backend runtimes have their own Python/CUDA requirements. |
| Python packages | [uv](https://docs.astral.sh/uv/getting-started/installation/), available on `PATH`. |
| Frontend | [Node.js with npm](https://nodejs.org/en/download); use a supported LTS release. The current Vite package requires Node 18 or Node 20+. |
| Tools | Git, Bash, and `make` for the shortcuts; `curl` for the checks below. |
| Real inference | NVIDIA driver with working `nvidia-smi`, a separately installed vLLM/SGLang runtime, and enough disk/VRAM for your model. |

Install uv if needed using its [official installer](https://docs.astral.sh/uv/getting-started/installation/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Open a new terminal if `uv` is not yet on `PATH`. Check your tools:

```bash
uv --version
node --version
npm --version
```

### 1. Get the repository and install dependencies

```bash
git clone https://github.com/mgravey/Shardon.git
cd Shardon
uv python install 3.11
UV_PYTHON=3.11 ./scripts/bootstrap.sh
make validate
```

Bootstrap installs all Python workspace packages and development tools into
`.venv`, installs the UI dependencies, and creates enabled-config symlinks.
It does **not** install NVIDIA drivers, vLLM/SGLang, or model weights.
You can choose another supported Python version by changing both `3.11` values.
`make setup` is the shortcut for the bootstrap script.

**Fresh installs only:** bootstrap's symlink-refresh step replaces enabled YAML
entries and enables **every** available definition, including demo definitions.
For an existing customized installation, use the [update procedure](docs/installation.md#updates-and-backups)
instead so your enabled selection is preserved.

### 2. Optional model-download settings

Skip this step for the mock demo. If your model needs Hugging Face access:

```bash
cp .env.example .env
chmod 600 .env
```

Do this only if you do not already have a `.env`. Edit it to set your real
`HF_TOKEN` and an **absolute, writable** `HF_HOME` cache directory. The example
`/var/lib/shardon/huggingface` may need to be created with the appropriate owner;
you can choose a directory under your home instead. The `.env` loader does not
expand `~` or `$HOME` inside values. Exported environment variables take precedence.

### 3. Start the services and sign in

```bash
make dev
```

This starts both APIs and the Vite development UI in the foreground. The script
is also available directly as `./scripts/run-local.sh`.

| Service | Default address | Purpose |
| --- | --- | --- |
| Admin UI | <http://127.0.0.1:5173> | Configure models, keys, GPUs, and inspect live activity/usage. |
| Admin API | <http://127.0.0.1:8081> | Control plane; API documentation at `/docs`. |
| Router API | <http://127.0.0.1:8080> | Inference plane; OpenAI client base URL ends in `/v1`. |

Open the UI to view loaded models, live activity, and the usage leaderboard without
signing in. To manage configuration or API keys, select **Admin sign in** and use
the demo credentials **`admin` / `admin`**.
[Set your own admin password](docs/installation.md#admin-password) before sharing
access. No inference API key is pre-created.

For a remote host, keep these loopback bindings and open an SSH tunnel from your
computer, then use the same local URLs:

```bash
ssh -N -L 5173:127.0.0.1:5173 -L 8080:127.0.0.1:8080 user@your-server
```

### 4. Create an API key and send a request

In the admin UI's API-key section, choose a **unique key ID** and an owner/user
name, create the key, and save the returned secret. Use distinct keys for different
applications to make the usage statistics useful.

In another terminal:

```bash
export SHARDON_API_KEY='paste-the-new-key-here'
curl --fail-with-body http://127.0.0.1:8080/health
curl --fail-with-body http://127.0.0.1:8081/health
curl --fail-with-body http://127.0.0.1:8080/v1/models \
  -H "Authorization: Bearer $SHARDON_API_KEY"
curl --fail-with-body http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer $SHARDON_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"demo-chat","messages":[{"role":"user","content":"Hello Shardon"}]}'
```

With the bundled config, the last request starts the mock backend and returns a
synthetic reply. With a real deployment, replace `demo-chat` with its
`api_model_name`; the first request can wait for model download/startup.
Health checks confirm the APIs are running; a successful inference request also
checks authentication, routing, backend startup, and response forwarding.

### 5. Check loaded models and usage

The large header names the loaded model(s). Below it, the live board shows who is
in flight, who is waiting, model-loading progress, GPU placement, and batch items.
The leaderboard counts requests and backend-reported tokens by API key or user.

Clients may request `"model":"currently-loaded"` to use a compatible model that
is already ready. If none is loaded, this returns 404; use a concrete model name
or manually load a deployment first. In the leaderboard, **Model names** switches
between **Actual model used** and **Requested model name**, which keeps those
requests grouped under `currently-loaded`.

The bundled mock replies do not report token usage or provide real SSE streaming:
request counts work, while token counts show **Unavailable**. This is expected.
Public statistics include owner names and key IDs, never key secrets. Configuration
and runtime controls still require admin authentication.
See [Activity and usage](docs/activity-and-usage.md) for coverage, history, and
streaming accounting details.

For configuration examples, separate service commands, password changes, startup
troubleshooting, deployment notes, backups, and upgrades, continue with the
[installation and operations guide](docs/installation.md).

## Core Ideas

- YAML is desired state.
- JSON and JSONL are observed state.
- Backends live in runtime folders and can be run with or without Shardon.
- GPU groups are first-class scheduling targets. Missing GPU observations block admission;
  UUID-pinned devices are never replaced by another GPU at the same PCI address.
- Deployment configuration supplies both GPU inventory and backend CUDA settings;
  inventory does not generate the backend environment. See [GPU mapping](docs/gpu-mapping.md)
  for precedence, per-group overrides, deployment-local files, and isolation limits.
- A deployment may have one or many eligible GPU groups; Shardon selects one concrete group per running process.
- Model capabilities are explicit metadata and are independent from request task routing.
- `keep_free` is enforced aggressively from observed process ownership.
- Drains are blocking runtime operations, not long-lived reservations.
- When a request needs another deployment on the same GPU group, idle loaded deployments can be evicted and unloaded first so memory can be reclaimed for the new load.
- Requests for unsupported/unconfigured models are rejected immediately and are not added to the interactive queue.

## Model Downloads

- `HF_TOKEN` is read from the process environment or from a repo-local `.env` file.
- The token is not stored in YAML, runtime JSON, or the admin UI.
- Backend subprocesses inherit `HF_TOKEN`, `HF_HOME`, and the model source information when launched.
- The admin UI now includes a guided model onboarding form and shows only whether `HF_TOKEN` is configured, never the secret itself.

## Backend Health Checks

- Shardon marks a backend healthy when the configured health endpoint returns a `2xx` response.
- JSON responses are stored as health payloads as-is.
- Non-JSON `2xx` responses (for example plain-text `/health` from some vLLM builds) are accepted and stored with status metadata instead of failing readiness.
- On startup and periodic health refresh, loaded runtime state is reconciled with live process IDs so stale loaded flags are cleared.
- A loaded deployment whose process dies or whose health endpoint becomes unreachable is stopped, marked unavailable, and remains eligible for normal on-demand loading.
- Connection-level request failures trigger one automatic cleanup, relaunch, readiness wait, and retry. Backend HTTP responses such as request-validation `4xx` errors do not restart the backend.
- Read, write, and connection-pool timeouts fail the request without restarting the backend; connection timeouts still permit recovery. Request failure diagnostics include the exception type.
- Streaming requests use the same recovery path only before their first response chunk; partial streams are never replayed automatically.
- Per-deployment lifecycle locks prevent concurrent requests and health checks from launching duplicate backend processes.
- Repeated identical backend health failures are de-duplicated in `state/events/events.jsonl` to avoid unbounded log spam during idle periods.
- Startup failures preserve the attempted group, log path, and known exit code in runtime diagnostics;
  operator log and event endpoints remain available after failure.
- Backend stop timeout is a maximum cap: if the process exits immediately after `SIGTERM`, Shardon proceeds without waiting for the full timeout window.
- Recovery diagnostics are written as `backend.health_failed`, `backend.stop`, `backend.start`, `backend.ready`, and `backend.recovery*` lifecycle events with deployment, GPU group, PID, reason, and result fields.

## Multimodal Routing

- Router supports OpenAI-style audio endpoints:
  - `POST /v1/audio/speech`
  - `POST /v1/audio/transcriptions`
  - `POST /v1/audio/translations`
- Transcription/translation accept multipart uploads and preserve text-style response formats (`text`, `srt`, `vtt`) at the API boundary.
- Router dependencies include `python-multipart` so a fresh `./scripts/bootstrap.sh` install can register multipart audio routes without manual pip installs.
- Backend adapters are modality-aware and include extension points for non-OpenAI-native backends (for example WhisperX) and future video operations without changing scheduler/runtime contracts.

## Runtime Operator Commands

- `shardon runtime status`
- `shardon runtime load --deployment <id>`
- `shardon runtime load --model <api-model>`
- `shardon runtime load --model <api-model> --gpu-group <group>`
- `shardon runtime unload --deployment <id>`
- `shardon runtime clear-queue [--batches]`

These commands are intended for cold-start debugging, manual health validation, and GPU-fit checks outside routed inference traffic.

Queue management notes:

- `clear-queue` clears interactive queued requests from runtime state.
- `clear-queue --batches` also cancels queued (not running) batch jobs.
- Admin API exposes the same operation at `POST /runtime/queue/clear`.

Runtime timeout defaults:

- `interactive_request_timeout_seconds` defaults to `300 + switch_grace_window_seconds` (600s with current defaults).
- `backend_startup_timeout_seconds` defaults to `300 + switch_grace_window_seconds` (600s with current defaults).
- `queue_poll_interval_seconds` controls how often queued interactive requests retry scheduling decisions while waiting for eviction/startup readiness.
- Backend YAML definitions can set `request_read_timeout_seconds` (default 600) and `request_connect_timeout_seconds` (default 10) for non-streaming JSON inference. Both also support GPU-group overrides; see [Configuration](docs/configuration.md#backend-inference-timeouts).

Chat requests forward optional `response_format` and `reasoning_effort` fields to the
backend. Supported reasoning values are `none`, `low`, `medium`, `high`, and `xhigh`;
the selected backend/model must support the requested options. Completed and failed
requests are removed from the queue in the same state update that finalizes them.

Live router smoke test:

- `SHARDON_API_KEY=<key> .venv/bin/python scripts/smoke_openai_surfaces.py`
- The script checks `/v1/completions`, streaming `/v1/completions`, `/v1/responses`, streaming `/v1/responses`, and a routing/switching pass. Use `--switch-model <model>` when the running config has a second text model to force a cross-model switch.
- Streaming checks expect the selected backend to support raw OpenAI-compatible SSE streaming.

## Multi-Group Deployments

- `deployments` can use `gpu_group_ids` as an ordered preference list; `gpu_group_id` remains supported for backward compatibility.
- Scheduler admission evaluates each eligible group in order and picks the first admissible group for the deployment.
- Runtime state records `selected_gpu_group_id` for the concrete running instance.
- Backends can define `gpu_group_overrides` for per-group `base_url`, `launch_command`, `environment`, and health/startup settings while keeping one backend definition.
- Only one process is ever launched per deployment id; switching groups unloads the existing process first.

## Documentation

- [Installation and operations](docs/installation.md)
- [Activity and usage](docs/activity-and-usage.md)
- [Architecture](docs/architecture.md)
- [Configuration](docs/configuration.md)
- [Scheduling](docs/scheduling.md)
- [GPU Mapping](docs/gpu-mapping.md)
- [Keep-Free](docs/keep-free.md)
- [Drain](docs/drain.md)
- [Batching](docs/batch.md)
- [Debugging](docs/debugging.md)
- [Running Backends Outside Shardon](docs/backends-outside-shardon.md)

## Live Activity and Token Usage

The public dashboard (no login required) prominently names every currently loaded model at the top,
with GPU placement and automatic refresh, even when activity filters are applied.
Below it, the live activity board shows requests by model,
API key, and owner. It distinguishes waiting, model loading, and backend in-flight
requests, shows wait reasons and GPU placement, and lists batch item progress.
Activity refreshes every two seconds while the page is visible; stale data stays
visible with a warning when an update fails.

The token leaderboard ranks API keys or users with per-model breakdowns for Today,
7 days, and 30 days (UTC calendar days). A Model names selector switches the
statistics between the actual model used and the requested name (including
`currently-loaded`). Counts come from backend-reported usage;
missing counts are shown as unavailable, with coverage indicating partial totals.
History is stored under `state/usage/`, survives restarts and key revocation, and
contains no prompts, responses, or API secrets. Tracking begins with this version.
Public statistics include owner names and key IDs, never key secrets. Configuration
and runtime controls still require admin authentication.
See [Activity and usage](docs/activity-and-usage.md) for API details and limitations.
