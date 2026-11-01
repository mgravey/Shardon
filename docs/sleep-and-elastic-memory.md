# Sleep Mode and Elastic Memory

Shardon can keep a vLLM process alive but asleep instead of stopping it. A
sleeping deployment keeps its process and port, frees almost all of its GPU
memory, and wakes in seconds instead of cold-loading for minutes.

Everything here is **opt-in**. Backends and deployments without a `sleep`
block, and GPU groups without `memory_model: elastic`, behave exactly as before.

## vLLM Sleep Mode

### Requirements

Start vLLM with `--enable-sleep-mode` and `VLLM_SERVER_DEV_MODE=1`. That
exposes these endpoints on the backend `base_url`:

| Endpoint | Purpose |
| --- | --- |
| `POST /sleep?level=1\|2&mode=wait\|abort` | Level 1 offloads weights to CPU RAM and discards the KV cache. Level 2 discards both. |
| `POST /wake_up[?tags=weights&tags=kv_cache]` | Re-allocate memory. |
| `POST /collective_rpc` `{"method":"reload_weights"}` | Level 2 only: reload weights between the two `wake_up` calls. |
| `GET /is_sleeping` | `{"is_sleeping": bool}`. Shardon confirms every transition with it. |

Only the control plane calls these endpoints. The router serves an explicit
list of OpenAI-compatible routes and never proxies `/sleep`, `/wake_up`,
`/collective_rpc`, `/is_sleeping` or any other dev endpoint (see
`tests/integration/test_sleep_api_surfaces.py`).

### Configuration

Put a `sleep` block on the backend. A deployment can override any field:

```yaml
# config/backends-available/vllm-stable-text.yaml
sleep:
  enabled: true
  level: 1                  # 1: weights to CPU RAM (fast wake); 2: discard weights
  idle_seconds: 600         # sleep after 10 minutes without requests (default)
  mode: wait                # wait for in-flight requests, or abort them
  sleep_timeout_seconds: 120
  wake_timeout_seconds: 120
  sleep_on_evict: true      # sleep instead of stop when another model needs the GPU
  max_sleep_seconds: 14400  # optional: stop completely after 4 hours asleep
  sleeping_memory_fraction: 0.05  # GPU memory a sleeping process keeps (estimate)

# config/deployments-available/qwen-big.yaml
sleep:
  idle_seconds: 300         # this deployment sleeps sooner
```

Defaults: `enabled: false`, `level: 1`, `idle_seconds: 600`, `mode: wait`,
both timeouts 120 s, `sleep_on_evict: false`, no `max_sleep_seconds`,
`sleeping_memory_fraction: 0.05`.

### States

| State | Loaded | Meaning |
| --- | --- | --- |
| `ready` | yes | Awake and serving. |
| `falling_asleep` | yes | `/sleep` in progress. |
| `sleeping` | yes | Same PID and port; `resident_memory_fraction` drops to the sleeping residual. The awake footprint is kept in `awake_memory_fraction`. |
| `waking` | yes | Wake sequence in progress. |
| `stopping` | yes | A sleeping deployment is being stopped by a sleep limit. |

These states appear in the admin and router status APIs, the public activity
view, the admin UI, and the event log.

### Idle Sleep

The router's background tick puts a deployment to sleep when all of these hold:

- it is `ready`;
- it has no active request and none queued for it;
- no batch job is queued for, or running on, it;
- nothing has happened for `idle_seconds`. Activity means a request was
  dispatched or finished, the deployment became ready, or it woke up.

Health and readiness probes (`health_path`, usually `/v1/models`) and the
router's own `/v1/models` listing are not activity, and never wake a deployment.

### Wake on Demand

Before an inference request or batch item reaches a sleeping deployment, Shardon
wakes it:

1. Waking needs the full footprint back, so it goes through the same admission
   planning as a cold load. In a static group, the other awake deployment is
   stopped, or put to sleep when it has `sleep_on_evict`.
2. One wake task per deployment is shared by every concurrent request.
3. Level 1 calls `wake_up`. Level 2 calls `wake_up?tags=weights`, then
   `collective_rpc reload_weights`, then `wake_up?tags=kv_cache`.
4. Shardon confirms with `/is_sleeping` before it forwards anything.
5. Wake time counts against the interactive queue timeout. A request that runs
   out of time fails with HTTP `409`, while the shared wake keeps going for the
   next request. Each wake is logged as `deployment.wake` with
   `duration_seconds`.
6. If the wake fails or exceeds `wake_timeout_seconds`, Shardon stops the
   process and cold-starts it (`deployment.wake_failed`, then
   `deployment.wake_fallback_restart`).

### Eviction

- `sleep_on_evict: true`: when a group needs memory for another deployment,
  Shardon puts this deployment to sleep instead of stopping it. That only
  happens when the incoming deployment uses a different host and port, because
  a sleeping process still holds its port. Deployments that share a backend
  `base_url` are always stopped.
- Sleeping deployments are always evicted (stopped) before awake ones, least
  recently used first.
- Static groups still run **one awake deployment at a time**. Sleepers may stay
  while memory allows.

### Limits: Avoiding a Pile of Sleepers

Level 1 parks weights in host RAM, so unlimited sleepers can exhaust RAM. Three
limits decide when a sleeping deployment is **stopped completely**:

| Setting | Scope | Effect |
| --- | --- | --- |
| `gpu_groups.<id>.max_sleeping` | per GPU group | Keep at most N sleepers in the group. |
| `max_sleeping_deployments` (`router.yaml`) | whole host | Keep at most N sleepers overall. |
| `sleep.max_sleep_seconds` | backend or deployment | Stop a deployment that has slept this long. |

When a cap is exceeded, Shardon stops the least recently used sleepers first and
logs `deployment.sleep_stop` with the reason. Limits are enforced after every
sleep and on every router tick. `max_sleeping: 0` makes deployments sleep only
transiently; prefer leaving `sleep.enabled` false instead.

### Controls

| Where | Control | Who |
| --- | --- | --- |
| Public activity board | **Sleep** button on an idle, sleep-enabled deployment | Anyone, no admin sign-in |
| Admin runtime table | **Sleep**, **Wake**, **Stop**, **Load** per deployment | Admin |
| `POST /runtime/activity/deployments/{id}/sleep` | Sleep (public) | No auth |
| `POST /runtime/sleep/{id}` and `/runtime/wake/{id}` | Sleep and wake | Admin token |
| `POST /runtime/unload/{id}` | Stop completely (works on sleeping deployments) | Admin token |

The public sleep endpoint is refused (HTTP `409`) while a deployment has active
or queued requests or batch work, or when sleep is not enabled for it. The next
request wakes it again, so it can only make a model momentarily slower. Public
sleeps are audited with the caller's address.

## Per-Deployment Ports

By default, every deployment of a backend shares the backend `base_url` port,
so two of them can never run at the same time. To co-host them, give the
backend a port range, or give a deployment a fixed port:

```yaml
# backend
base_url: http://127.0.0.1:8100   # host and path are kept; the port is replaced
port_range: [8100, 8109]          # inclusive

# deployment (optional; wins over the backend range)
port: 8120
```

- At start, Shardon allocates the lowest port in the range that no loaded or
  starting deployment holds and that is not already bound. The allocation is
  made inside the state-file lock, so the admin and router processes never hand
  out the same port.
- The port is passed to the launch command as `IGF_SHARDON_RUNTIME_PORT` (also
  `SHARDON_RUNTIME_PORT`). Your launcher must start vLLM on it, for example
  `--port "$IGF_SHARDON_RUNTIME_PORT"`.
- Shardon targets the deployment's own `base_url` for readiness, health,
  inference and sleep control.
- The port is released when the deployment stops or fails to start. A sleeping
  deployment keeps its port.
- An exhausted range fails the start with `no free port in port_range`.
- Backends without `port_range` or `port` keep working exactly as before.

## Elastic Groups (kvcached Co-Residency)

kvcached makes the KV cache elastic: weights stay loaded, and KV pages are
mapped on demand and freed after requests finish. In an `elastic` GPU group,
several deployments stay resident together:

```yaml
# gpu group
memory_model: elastic        # default: static (one awake deployment at a time)
usable_memory_fraction: 0.95
```

### Admission

Each loaded deployment has a resident footprint, `resident_memory_fraction`:
its weights plus its kvcached reserve.

- **Before observation**, Shardon uses a footprint learned on an earlier run in
  the same group, else the deployment's `resident_memory_fraction`, else its
  `memory_fraction`.
- **After readiness**, Shardon sums per-PID `nvidia-smi` memory over the
  deployment's process tree (vLLM allocates in child processes). It adds the
  kvcached reserve: the per-instance `kvcached.limit` or `limit_percent`, or 0
  without one. It retries on every router tick until it has an observation, and
  logs `deployment.footprint_observed`.

A deployment is admitted when both of these hold:

1. **Accounting**: the footprints of everything that stays, plus the new one,
   fit within `usable_memory_fraction`.
2. **Physical**: observed free memory on each GPU, plus what the plan
   reclaims, covers the new footprint while keeping `1 - usable_memory_fraction`
   free. Processes outside Shardon, such as WhisperX, TTS or Voicebox, are
   counted here.

Loaded deployments are **not** evicted when the new one fits. When it does not
fit, memory is freed in this order:

1. sleep idle `sleep_on_evict` deployments, least recently used first;
2. stop sleeping deployments, least recently used first;
3. stop idle awake deployments, least recently used first.

Busy deployments are never touched. If the new deployment still does not fit,
the request waits in the queue as usual. Waking a sleeping deployment goes
through the same planning. `keep_free` still stops every Shardon deployment in
the group when another user appears on the GPU.

### kvcached Controls

```yaml
kvcached:                 # backend and/or deployment; deployment fields win
  enabled: true           # implied by any kvcached block or ENABLE_KVCACHED=true
  limit: 16G              # kvctl limit <ipc> 16G   (or limit_percent: 20)
  min: 4G                 # kvctl min <ipc> 4G
  priority: 10            # kvctl priority <ipc> 10
  headroom: 2G            # kvctl headroom <ipc> 2G
  kvctl_path: /opt/igf/vllm-stable/.venv/bin/kvctl   # default: kvctl on PATH
  ipc_name: null          # default: igf-shardon-<deployment id>
```

- `KVCACHED_IPC_NAME` defaults to `igf-shardon-<deployment id>` and is passed to
  the launch command. A `KVCACHED_IPC_NAME` set in the backend `environment` is
  kept unchanged.
- Shardon applies the controls with `kvctl` after readiness, because `kvctl`
  ships in vLLM's environment rather than Shardon's. A failure is logged as
  `kvcached.apply_failed` and never fails the deployment.
- The router tick and both status endpoints poll `kvctl list <ipc...> --json`.
  Live usage (`used_bytes`, `limit_bytes`) appears in `kvcached_usage` in the
  status APIs, in the admin runtime table and on the activity board.

## Example: TW Node

One RTX 6000 Pro (96 GB) shared with WhisperX, a TTS engine and Voicebox
(outside Shardon). The host has 576 GB RAM for level-1 sleepers.

```yaml
# config/gpu-groups-available/rtx6000.yaml
id: rtx6000
display_name: RTX 6000 Pro
gpu_ids: [rtx6000-0]
memory_model: elastic
usable_memory_fraction: 0.80   # Shardon's budget; the rest stays for the other services
max_sleeping: 2

# config/backends-available/vllm-stable-text.yaml
id: vllm-stable-text
backend_type: vllm
version: 0.30.0
display_name: vLLM 0.30 + kvcached
runtime_dir: /opt/igf/vllm-stable
base_url: http://127.0.0.1:8100
port_range: [8100, 8109]
launch_command: [/opt/igf/vllm-stable/launch.sh]  # vllm serve ... --port "$IGF_SHARDON_RUNTIME_PORT" --enable-sleep-mode
health_path: /v1/models
environment:
  VLLM_SERVER_DEV_MODE: "1"
  ENABLE_KVCACHED: "true"
  KVCACHED_AUTOPATCH: "1"
sleep:
  enabled: true
  level: 1
  idle_seconds: 600
  sleep_on_evict: true
  max_sleep_seconds: 21600   # fully stop after 6 hours asleep
kvcached:
  kvctl_path: /opt/igf/vllm-stable/.venv/bin/kvctl

# config/deployments-available/qwen-text.yaml (excerpt)
backend_runtime_id: vllm-stable-text
gpu_group_id: rtx6000
memory_fraction: 0.45
resident_memory_fraction: 0.35   # first-load estimate; replaced by the observed footprint
kvcached:
  limit: 16G
  priority: 10

# config/router.yaml (excerpt)
max_sleeping_deployments: 3
```

Migrating from the current single-port setup: add `port_range` and make the
launcher honour `IGF_SHARDON_RUNTIME_PORT`. Then switch the group to `elastic`.
Until the group is switched, it keeps its `static` behaviour.
