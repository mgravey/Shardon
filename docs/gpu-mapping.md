# GPU Mapping

Shardon uses configured GPU IDs such as `worker-a` as scheduling identities.
The NVIDIA provider maps observations to those IDs by UUID, independently of
probe order or CUDA device indices. Use full GPU UUIDs in deployment configuration.

## Identity and availability

- A configured UUID is authoritative. If it disappears, Shardon does not substitute
  a GPU at its old PCI address, even when both identity fields are configured.
- PCI bus ID is used only when no UUID is configured. A PCI-only definition tracks
  a slot and can therefore match replacement hardware; use UUIDs to pin a device.
- `cuda_visible_index` is metadata, not an identity lookup or a launch setting.
- Each inventory refresh replaces previous observations. Missing devices and
  failed probes cannot retain stale memory availability.
- Every device in a nonempty GPU group must be observed before the scheduler
  admits work, including reuse of a loaded deployment. Startup rechecks the
  selected group before launching or recovering its backend.
- `gpu_group_ids` is an explicit, ordered list of allowed placements. An unavailable
  preferred group may yield to another **configured eligible group**. A manual
  `--gpu-group <group>` selection is pinned and fails if unavailable. Startup never
  retries on another group when its selected devices disappear after admission.

Availability is a point-in-time observation. A successful `nvidia-smi` probe does
not prove CUDA initialization will succeed in the backend's environment, and a
GPU can disappear after preflight. Readiness and backend logs diagnose those cases.
Shardon does not migrate or terminate already in-flight work merely because a
subsequent probe is missing. Empty groups retain their existing no-GPU admission
behavior. `gpu_provider` in `config/router.yaml` defaults to `nvidia-smi`. The
shipped demo explicitly uses `gpu_provider: mock` for synthetic observations;
**set `gpu_provider: nvidia-smi` for real inference**. Mock observations do not
validate hardware availability. Restart both services after changing the provider.

## Deployment-supplied configuration

GPU inventory controls scheduling and observation. It does **not** generate,
rewrite, or validate `CUDA_VISIBLE_DEVICES` in the backend environment. The operator
must align each group's inventory with its backend settings, including runtime
paths, ports, engine flags, and tensor-parallel size. Backends remain independently
runnable with the same environment and command outside Shardon.

Keep actual UUIDs, PCI addresses, paths, and private runtime settings outside the
tracked repository. Use a separate deployment root containing `config/` and
`state/`, supplied through `SHARDON_REPO_ROOT` to both services and the CLI. If using
a checkout for deployment, `config/**/*.local.yaml` files and enabled links with
the same suffix are ignored. These are ordinary complete YAML definitions, not
an overlay mechanism: enable exactly one definition per ID and disable conflicting
demo links. The tracked examples below contain placeholders only; replace them
in deployment-local files before use.

```yaml
# config/gpu-inventory-available/worker-a.local.yaml
id: worker-a
uuid: "GPU-<FULL-UUID-A>"
```

```yaml
# config/gpu-groups-available/group-a.local.yaml
id: group-a
display_name: Primary inference group
gpu_ids: [worker-a]
```

In the deployment definition, select a single `gpu_group_id: group-a`, or explicitly
allow alternatives with `gpu_group_ids: [group-a, group-b]`. Each referenced group
needs its own inventory definition. Relevant fields in a backend definition:

```yaml
environment:
  CUDA_VISIBLE_DEVICES: "GPU-<FULL-UUID-A>"
  CUDA_DEVICE_ORDER: "PCI_BUS_ID"
gpu_group_overrides:
  group-b:
    base_url: "http://127.0.0.1:<GROUP-B-PORT>"
    launch_command: ["<RUNTIME-LAUNCHER>", "--port", "<GROUP-B-PORT>"]
    working_directory: "<RUNTIME-DIRECTORY>"
    environment:
      CUDA_VISIBLE_DEVICES: "GPU-<FULL-UUID-B>"
```

Define overrides for **every eligible group whose runtime settings differ**.
An omitted override inherits the base backend settings; Shardon does not infer a
new CUDA selection from the chosen group. The selected group is passed to launch,
readiness, inference adapter resolution, and stop operations, and recorded in
`selected_gpu_group_id` in runtime state.

For multi-GPU execution, set an ordered UUID list such as
`GPU-<FULL-UUID-A>,GPU-<FULL-UUID-B>` and configure the engine for that device count.
CUDA enumerates visible devices in the supplied order, so backend-local ordinal
zero refers to the first selected GPU, not necessarily host ordinal zero.
`CUDA_DEVICE_ORDER` does not pin an ordinal to a physical GPU when availability
changes. See [NVIDIA's CUDA environment variable reference](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html).

## Environment precedence

Values are merged in this order, with later layers winning per key:

1. The Shardon service's process environment. Service/CLI startup loads its
   deployment-root `.env` only for keys not already present in the environment.
2. The backend's base `environment` dictionary.
3. The selected group's `gpu_group_overrides.<group>.environment` dictionary.
4. Shardon's generated launch metadata, including `SHARDON_GPU_GROUP_ID`,
   `SHARDON_DEPLOYMENT_ID`, and model information.

CUDA variables are not generated metadata. Configured values, including UUID
lists, are passed unchanged to the subprocess. When omitted, a variable is
inherited; an explicit empty string overrides an inherited value. YAML environment
values are literal strings: `$VARIABLE` and `~` are not expanded by Shardon.
Launch wrappers can still alter their child's environment; verify those separately.
Keep secrets in the service environment or `.env`, not backend YAML exposed in
admin configuration. Restart both Shardon services after deployment YAML changes.

## Diagnostics and isolation limits

Admission failures name unavailable configured GPU IDs and groups. Runtime group
status includes `unavailable_gpu_ids`. Probe execution/parsing errors appear in
service logs. See [Debugging](debugging.md) for startup events, exit codes, and
backend stdout/stderr access.

CUDA device selection is a cooperative runtime setting, not enforced device
isolation. It does not establish exclusive ownership, block another process from
accessing devices, or prove passthrough safety. Inventory, `keep_free`, and readiness
checks likewise do not establish that a device is safe for passthrough. Device
isolation and passthrough lifecycle coordination belong to deployment infrastructure;
Shardon does not manage drivers, VFIO, or systemd device restrictions.
