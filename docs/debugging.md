# Debugging

Shardon is designed for practical local operations.

## Useful Surfaces

- `state/runtime.json`: current observed snapshot
- `state/events/events.jsonl`: lifecycle and scheduling events
- `state/audit/audit.jsonl`: admin and key actions
- `state/logs/<deployment>.log`: supervised backend stdout and stderr
- `GET /runtime/logs/{deployment_id}`: control-plane log access
- `GET /runtime/events`: recent event stream
- `POST /runtime/load/{deployment_id}`: manually load a deployment
- `POST /runtime/unload/{deployment_id}`: manually unload a deployment
- `shardon runtime status|load|unload`: operator CLI

Backend runtime folders are intentionally runnable outside Shardon for direct troubleshooting.

## GPU admission and startup failures

When admission reports `unavailable GPUs (missing observations)`, check the named
inventory IDs and their full UUIDs against the deployment's visible hardware.
A failed inventory probe makes devices unavailable; its error is in the Shardon
service logs. `unavailable_gpu_ids` in group status identifies missing observations.
A UUID-pinned device is never replaced by another GPU at the same PCI address.
See [GPU mapping](gpu-mapping.md) for explicit fallback groups and environment precedence.

Backend failures retain the attempted group and diagnostic details in
`state/runtime.json` (`last_error` and `last_readiness_detail`):

- `backend.start_failed`: GPU preflight failure or a launch exception, such as a
  missing executable or invalid working directory. Launch exceptions include the
  exception type and intended `log_path`; a filesystem error may prevent creation
  of that log file.
- `backend.readiness_failed`: early process exit or readiness timeout. Includes
  `log_path` and the natural `returncode` when known; it is null for a process still
  running at timeout or an adopted process whose exit code cannot be recovered.

Backend stdout and stderr are combined and appended to
`state/logs/<deployment_id>.log`, including across retries. The authenticated admin
endpoints `GET /runtime/logs/{deployment_id}` and `GET /runtime/events` provide log
and event tails even after failed startup. Non-UTF-8 log bytes are replaced for
API display; the original log remains intact. Startup failures return structured
HTTP 409 diagnostics from manual load operations instead of leaving a deployment
stuck in `starting`. Do not put secrets in backend output.
