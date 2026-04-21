# Shardon Repository Conventions

## Mission

Shardon is a Linux-first self-hosted LLM router and admin platform for constrained GPU environments.
The guiding idea is dynamic orchestration: load models when needed, allow coexistence when memory budgets permit, and keep the control plane separate from inference.

Tagline: `Shardon — pardon, one model at a time.`

## Monorepo Layout

- `apps/admin_api`: FastAPI admin/control plane.
- `apps/router_api`: FastAPI inference/router plane.
- `apps/admin_web`: React/Vite admin UI.
- `packages/shardon_core`: Shared Python domain logic, schemas, scheduling, state, adapters, and services.
- `config`: YAML desired state with `available/` and `enabled/` folders.
- `state`: JSON/JSONL observed runtime state. Treat as API-managed.
- `demo`: Independently runnable mock backend runtimes.
- `docs`: Operational and architecture documentation.
- `tests`: End-to-end and integration coverage.

## Architectural Rules

- Keep admin and router APIs fully separate.
- Static desired state belongs in YAML under `config/`.
- Runtime or observed state belongs in JSON or JSONL under `state/`.
- Do not introduce a database.
- Backend runtimes must remain independently runnable outside Shardon.
- Scheduler decisions should use stable GPU identities, not transient CUDA indices.
- Prefer explicit typed schemas over ad-hoc dictionaries.

## Python Guidelines

- Target Python 3.11+.
- Use FastAPI for both services.
- Use Pydantic models for config and API schemas.
- Shared logic must live in `packages/shardon_core` rather than inside apps.
- File writes must be atomic and lock-protected.
- Process supervision must preserve logs and crash metadata for local debugging.

## Frontend Guidelines

- Keep the admin UI operationally focused.
- Favor fast scanning, clear status colors, and dense but readable layouts.
- Treat the UI as a control surface for a local operator rather than a marketing site.

## Config and State Conventions

- `*-available/` contains canonical definitions.
- `*-enabled/` is the active set and should be symlink-friendly.
- API keys and admin credentials persist as hashed material only.
- Runtime events should be appended to JSONL for auditability.

## Testing Expectations

- Unit tests for shared logic live in `packages/shardon_core/tests`.
- Integration tests for service behavior live in `tests/integration`.
- Use mock GPU providers and mock backend runtimes for deterministic tests.

