# Scheduling

Shardon schedules both interactive and batch work against deployments rather than raw model names.

## Interactive Requests

- Requests may queue while Shardon waits for a compatible deployment to become admissible.
- If a compatible deployment is already loaded, reuse is preferred.
- If switching is needed, LRU is the default eviction strategy.
- If lower-priority work is still queued for the currently loaded model, Shardon preserves that model for up to five minutes before switching.
- When no admissible deployment exists, the router returns HTTP `409`.

## Batch Jobs

- Batch jobs use an OpenAI-compatible submission surface.
- Batch execution is handled by Shardon itself.
- Batch work never forces a model switch on a busy group.
- Extra Shardon endpoints expose progress beyond the standard batch API.

