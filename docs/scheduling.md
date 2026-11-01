# Scheduling

Shardon schedules both interactive and batch work against deployments rather than raw model names.

## Interactive Requests

- Requests may queue while Shardon waits for a compatible deployment to become admissible.
- If a compatible deployment is already loaded, reuse is preferred.
- For deployments with multiple eligible GPU groups, scheduler evaluates groups in declared order and selects the first admissible target.
- If switching is needed, LRU is the default eviction strategy.
- If lower-priority work is still queued for the currently loaded model, Shardon preserves that model for up to five minutes before switching.
- When a target GPU group is occupied by another managed deployment, Shardon evaluates whether it can evict and swap that deployment cleanly.
- Backend startup is gated by readiness checks before the request is forwarded.
- GPU groups default to `memory_model: static` (one awake deployment at a time). `elastic` groups co-host deployments while their resident footprints (weights plus kvcached reserve, learned from per-process GPU memory) fit both the budget and observed free memory. When something must give way, idle `sleep_on_evict` deployments are slept first, then the least recently used deployments are stopped.
- Opt-in vLLM sleep mode: a sleeping deployment stays loaded (same process and port) and is woken before any request is forwarded. Waking re-runs admission like a load. Sleeping deployments are evicted before awake ones, and `sleep_on_evict` deployments are put to sleep instead of stopped when the incoming deployment uses another port. See [Sleep mode and elastic memory](sleep-and-elastic-memory.md).
- When no admissible deployment exists, the router returns HTTP `409` with candidate and GPU-group diagnostics instead of only a generic timeout.

## Batch Jobs

- Batch jobs use an OpenAI-compatible submission surface.
- Batch execution is handled by Shardon itself.
- Batch work never forces a model switch on a busy group, and never displaces an awake deployment.
- Batch work wakes a sleeping deployment before its first item. Queued or running batch work keeps its deployment from idle or manual sleep.
- Extra Shardon endpoints expose progress beyond the standard batch API.
