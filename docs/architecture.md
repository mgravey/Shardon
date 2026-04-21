# Architecture

Shardon is organized as a single monorepo with two distinct FastAPI services and one shared Python core.

## Planes

- Admin plane: login, config CRUD, key management, drain operations, validation, queue inspection, events, and logs.
- Router plane: OpenAI-compatible inference surface, scheduling, queueing, backend selection, model loading, and batch execution.

## Shared Core

- Config loader: reads YAML desired state from `config/`.
- Runtime state store: persists JSON snapshots and JSONL events under `state/`.
- Scheduler: selects deployments using compatibility, drain state, memory budgets, group policies, and grace-window switching logic.
- Backend registry: starts and stops runtime folders directly and proxies requests to backend HTTP APIs.
- GPU provider: abstracts observations so NVIDIA tooling is an implementation detail rather than a global assumption.

## Runtime Model

- Models are logical metadata.
- Deployments are concrete runnable placements bound to backend runtime plus GPU group plus memory policy.
- Multiple deployments may coexist on one GPU group if budget policy allows it.
- Backends remain independently runnable from their own runtime directories.

