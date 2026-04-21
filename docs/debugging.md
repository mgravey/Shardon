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
