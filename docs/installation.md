# Installation and operations

Start with [Install and First Run in the README](../README.md#install-and-first-run)
for prerequisites, a fresh checkout, login, key creation, and the first request.
This guide covers machine-specific configuration and ongoing operation.
All relative commands below assume you are in the repository root.

## What setup installs

`./scripts/bootstrap.sh` runs these steps:

1. `uv sync --all-packages --group dev` installs the shared package, admin API,
   router API, and development dependencies into the workspace environment.
2. `npm install` installs the React/Vite UI dependencies.
3. `./scripts/refresh_enabled_symlinks.sh` recreates enabled YAML entries for
   **all** definitions in the corresponding available directories.

The third step removes existing enabled YAML files/symlinks before recreating
links. Use bootstrap on a fresh clone. For later dependency updates, run the
first two commands individually and preserve your enabled selection.

Python 3.11+ is supported, including 3.14. Shardon's dependency minimums include
FastAPI 0.118.3, Pydantic 2.12, Uvicorn 0.38, and PyYAML 6.0.3. Inference engines
are separate installations with their own compatibility requirements. No database,
GPU driver, inference engine, or model weights are installed by bootstrap.

## Admin password

The bundled account is `admin` with password `admin`. Stop the admin service
before changing its configuration. This command prompts without echoing your
new password and persists only its hash:

```bash
uv run --all-packages python - <<'PY'
from getpass import getpass
from pathlib import Path
import yaml
from shardon_core.auth.service import hash_secret
from shardon_core.utils.files import atomic_write_text, locked_file

password = getpass("New admin password: ")
if not password or password != getpass("Confirm password: "):
    raise SystemExit("Password is empty or confirmation does not match")
path = Path("config/auth/admins-available/admin.yaml")
with locked_file(path.with_suffix(".lock")):
    account = yaml.safe_load(path.read_text())
    account["password_hash"] = hash_secret(password)
    atomic_write_text(path, yaml.safe_dump(account, sort_keys=False))
print("Admin password updated. Restart the admin service.")
PY
```

Keep the enabled admin entry pointing to this canonical file. New login attempts
use the new password after restart. This procedure does not revoke already issued
admin tokens; keep existing sessions in mind when changing access.

For inference, create a key in the UI and copy the secret when shown. A key ID
identifies one application; its owner field identifies the user. Choose a new,
unique ID for each key: creating a key with an existing ID replaces that record.
Admin login tokens and inference API keys are different credentials.

## Real GPU and backend setup

### 1. Map the actual GPUs

On the Linux inference host:

```bash
nvidia-smi
nvidia-smi --query-gpu=uuid,pci.bus_id,name,memory.total --format=csv
```

Replace demo UUIDs and PCI bus IDs under `config/gpu-inventory-available/` with
values from your machine. Keep identifiers such as `gpu0` stable; use them in
`gpu_ids` under `config/gpu-groups-available/`.

Enable only the inventory, groups, and deployments you intend to use. On a
one-GPU machine, remove the enabled links for the unused demo GPU/group and any
deployments referring to them. Remove only enabled links to disable definitions;
keep canonical files in the available directories for later reuse.

The supplied `group-b` has `keep_free: true`: another OS user's GPU process causes
Shardon to stop its own workloads on that group. Choose this policy deliberately
for shared hardware; see [Keep-Free](keep-free.md).

GPU inventory is used for scheduling and observation. It does **not** automatically
set `CUDA_VISIBLE_DEVICES` for the subprocess. Set that variable in backend
`environment` or `gpu_group_overrides`, matching the group's GPU UUIDs. For a
multi-GPU backend, also configure the engine's tensor parallelism appropriately.
See [GPU mapping](gpu-mapping.md) and [Configuration](configuration.md).

### 2. Install and verify the inference engine separately

Choose a runtime version compatible with your GPU, driver, Python, model, and
required endpoints. Install it in its own environment using the engine's guide:
[vLLM installation](https://docs.vllm.ai/en/stable/getting_started/installation/)
or [SGLang installation](https://docs.sglang.io/docs/get-started/install).
The version strings in the bundled **mock** backend YAML describe demo labels;
they do not install those engine versions.

Use absolute paths for the backend executable and working directory. The account
running Shardon must be able to execute the runtime, read the model weights,
write the model cache, and write Shardon's `state/` directory. Run both Shardon
planes under the same service account and against the same checkout/state root.

The following example assumes you have installed vLLM under
`/srv/shardon-backends/vllm/.venv`. Create
`/srv/shardon-backends/vllm/launch.sh` with this content and make it executable:

```bash
#!/usr/bin/env bash
set -euo pipefail
: "${SHARDON_MODEL_SOURCE:?Set SHARDON_MODEL_SOURCE}"
: "${SHARDON_API_MODEL_NAME:?Set SHARDON_API_MODEL_NAME}"
exec /srv/shardon-backends/vllm/.venv/bin/vllm serve "$SHARDON_MODEL_SOURCE" \
  --served-model-name "$SHARDON_API_MODEL_NAME" \
  --host 127.0.0.1 --port 19101 "$@"
```

The launcher's positional model and served name must match the model source and
API alias. Consult [vLLM online serving](https://docs.vllm.ai/en/stable/serving/online_serving/)
for the server options supported by your installed version. Shardon appends
model-specific runtime arguments to the launcher; `"$@"` passes them through.
Use an equivalent independently runnable launcher for SGLang.

### 3. Register the backend, model, and deployment

These are templates. Replace the example GPU UUID, engine version, paths, and
`YOUR_ORG/YOUR_MODEL` source before using them. The alias `my-chat` is the name
clients will send; the internal model ID is `my-model`.

Create `config/backends-available/local-vllm.yaml`:

```yaml
id: local-vllm
backend_type: vllm
version: "YOUR_INSTALLED_VERSION"
display_name: Local vLLM
runtime_dir: /srv/shardon-backends/vllm
working_directory: /srv/shardon-backends/vllm
base_url: http://127.0.0.1:19101
launch_command:
  - /srv/shardon-backends/vllm/launch.sh
health_path: /health
environment:
  CUDA_VISIBLE_DEVICES: "GPU-REPLACE-WITH-YOUR-UUID"
capabilities:
  modalities: [text]
  chat: true
  responses: false
  completions: true
  embeddings: false
  batch: true
  openai_compatible: true
```

This example enables chat and completions. Enable `responses` and add the
corresponding tasks below only if your engine version/model supports that endpoint.

Create `config/models-available/my-model.yaml`:

```yaml
id: my-model
source: YOUR_ORG/YOUR_MODEL
display_name: My Chat Model
backend_compatibility: [vllm]
tasks: [chat, completion]
model_capabilities: [text]
runtime_launch_args:
  - --gpu-memory-utilization
  - "0.85"
```

Create `config/deployments-available/my-chat.yaml`:

```yaml
id: my-chat
model_id: my-model
backend_runtime_id: local-vllm
gpu_group_id: group-a
api_model_name: my-chat
display_name: My Chat / Group A
memory_fraction: 0.85
enabled: true
tasks: [chat, completion]
```

Choose memory settings that fit your hardware and model. Shardon's
`memory_fraction` is a scheduling budget; it does not configure the engine's
memory allocation. Keep both settings consistent. Each simultaneously runnable
backend process needs a distinct listening port; the launch port must match
`base_url`. Multi-group backends may use per-group overrides for ports and devices.

Enable just these definitions, preserving any other definitions you still need:

```bash
ln -s ../backends-available/local-vllm.yaml config/backends-enabled/local-vllm.yaml
ln -s ../models-available/my-model.yaml config/models-enabled/my-model.yaml
ln -s ../deployments-available/my-chat.yaml config/deployments-enabled/my-chat.yaml
make validate
```

The links must not already exist when using the commands above. Disable unused
demo deployment links so clients do not accidentally select mock models.
Restart both APIs after editing configuration, including edits made through the
admin UI, so both processes use the same definitions. `make validate` checks YAML
loading and schema validity; it does not download weights, launch engines, or
prove that the model fits in VRAM.

Once the backend and GPU groups exist, the UI's **Add Model** and deployment forms
can also create the model/deployment definitions. They do not install the engine.

### 4. Load the model and verify inference

With the services running, use the UI's load action or:

```bash
.venv/bin/shardon runtime load --deployment my-chat
curl --fail-with-body http://127.0.0.1:19101/health
curl --fail-with-body http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer $SHARDON_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"my-chat","messages":[{"role":"user","content":"Hello"}]}'
```

Check `state/logs/my-chat.log` during cold startup. `HF_TOKEN` and `HF_HOME` come
from Shardon's environment or repo `.env` and are inherited by backend processes.
Models requiring access approval must be accessible to that token.

After loading, `currently-loaded` can be used as the requested model name.
If several compatible deployments are ready, Shardon chooses the most recently
used/loaded candidate (with deployment ID as the tie breaker); use a concrete
alias when you need a specific model. Usage stores both the requested name and
the resolved name; the leaderboard selector changes which one is grouped.

## Starting services and choosing ports

`make dev` is the convenient local foreground launcher. For separate terminals:

```bash
make admin
```

```bash
make router
```

```bash
make web
```

Their underlying commands are `uv run --package shardon-admin-api shardon-admin-api`,
`uv run --package shardon-router-api shardon-router-api`, and
`npm --workspace apps/admin_web run dev`.

API bind addresses/ports are configured in `config/router.yaml`. The UI's port
can be changed with an exported variable:

```bash
SHARDON_ADMIN_UI_PORT=5174 make web
```

If the admin API port changes, also update the `/api` proxy target in
`apps/admin_web/vite.config.ts`. The Python `.env` loader applies to API/CLI
processes; export the UI port variable in the shell for the Vite configuration.
Set `SHARDON_REPO_ROOT` to an absolute checkout path when invoking the CLI from
another directory. Backend relative paths still require an appropriate working
directory; absolute runtime paths are preferable on servers.

`Ctrl+C` stops the foreground launcher and its services. Do not assume that it
unloads every backend subprocess: use the UI unload action or
`.venv/bin/shardon runtime unload --deployment my-chat` before shutting down when
you need to release GPU memory.

For unattended operation, supervise the two API entrypoints separately using your
service manager, with the checkout as their working directory and the same
`SHARDON_REPO_ROOT`, environment, and service account. Keep a single process per
plane; multi-worker/multi-router orchestration is not configured by these scripts.
No systemd units, TLS endpoint, or public-hosting setup is installed by bootstrap.

Build static UI assets with `npm run build:web`; the output is
`apps/admin_web/dist/`. When serving that build, configure your web server to
proxy `/api/...` to the admin API with `/api` stripped. The frontend uses this
relative prefix; a static file server alone cannot provide admin API access.
Keep the inference plane separate at the router's `/v1` base URL. The development
Vite server already implements the admin proxy for local use.

## Updates and backups

Unload models and stop the services for a consistent backup. Preserve `config/`,
`state/`, `.env` if present, your runtime launchers, and the backend environment
installation details. `state/auth/` contains hashed keys and the admin signing
secret; `state/usage/` holds usage history. State and `.env` are ignored by Git,
so pushing the repository is not a backup.

One local backup example, run from the checkout after stopping services:

```bash
umask 077
mkdir -p backups
tar -czf "backups/shardon-$(date +%Y%m%d-%H%M%S).tar.gz" config state
```

Include `.env` in the archive's file list if it exists. Store another copy outside
the checkout; the local `backups/` directory is ignored by Git.

Review and preserve local configuration/code changes before pulling updates:

```bash
git status --short
git pull --ff-only
uv sync --all-packages --group dev
npm install
make validate
npm run build:web
```

Restart both API processes and the UI/server, then check both `/health` endpoints
and send an inference request. Do not rerun bootstrap or the symlink-refresh script
as an upgrade step unless you intend to enable every available definition.
After an OS/Python upgrade, recreate `.venv` using your selected interpreter and
reinstall workspace dependencies; keep inference environments separate. Restored
runtime process IDs may be stale, so inspect status and explicitly load the needed
models rather than assuming restored JSON means inference is ready.

## Troubleshooting and verification

| Symptom | Check |
| --- | --- |
| `uv` or `npm` not found | Install the prerequisite, reopen the shell, and check `PATH`. |
| No models or missing definitions | Inspect the enabled symlinks, run `make validate`, and restart both planes. |
| Admin UI loads but API calls fail | Check admin `/health`, credentials, and the UI `/api` proxy target. |
| Login fails | Fresh demo credentials are `admin` / `admin`; restart after editing the admin YAML. |
| Inference returns 401 | Use an inference API key, not an admin token; check key revocation. |
| `currently-loaded` returns 404 | No compatible model is ready; load one or request a concrete alias. |
| Request waits or startup times out | Check the live wait reason and backend log, GPU placement, memory, model access, and launch command. |
| Empty GPU observations | Check `nvidia-smi` and configured UUID/PCI identities. Demo inventory contains placeholders. |
| Backend exits or uses the wrong GPU | Verify executable permissions, environment, `CUDA_VISIBLE_DEVICES`, port availability, and engine/CUDA compatibility. |
| Hugging Face download fails | Check token access, model approval, network, and writable absolute cache path. |
| Token counts say Unavailable | Backend did not report usage; this is normal for bundled mocks and some audio/streaming responses. |
| Dashboard shows Stale | Last good data is retained; check connectivity, admin token expiry, and API health. |
| Port already in use | Check API/UI configuration and backend launch ports; avoid manually starting a second copy. |

The default queue/startup timeouts are 600 seconds with the shipped configuration.
See [Configuration](configuration.md#backend-inference-timeouts) for backend HTTP
timeouts and [Debugging](debugging.md) for event and log locations.
`shardon runtime status` performs health/enforcement work; the admin
`/runtime/activity` and `/runtime/usage` endpoints are read-only views.

Development verification:

```bash
make test
npm run build:web
```

For a configured real backend supporting both Responses and Completions streaming:

```bash
SHARDON_API_KEY='your-key' .venv/bin/python scripts/smoke_openai_surfaces.py \
  --model my-chat --timeout 600
```

This smoke check is broader than the first-request example: enable the relevant
backend capabilities/model tasks first. Use `--switch-model other-alias` when a
second compatible model is configured. Bundled mock runtimes do not implement
real SSE streaming, so they are not a passing target for the streaming checks.
