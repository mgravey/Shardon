# Batch Behavior

Shardon accepts batches through an OpenAI-compatible endpoint and executes them internally.

## Rules

- Batch jobs queue in file-backed runtime state.
- The scheduler only runs batch work when a matching deployment is already loaded or an idle group can accept it.
- Batch jobs do not evict interactive workloads.
- Progress is available through Shardon-specific endpoints because the standard OpenAI batch surface is not sufficient for local operations.

