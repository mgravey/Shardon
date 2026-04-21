# Keep-Free

`keep_free` is a GPU-group policy, not a reservation object.

When enabled:

- Shardon watches GPU processes on the group.
- Router-managed backend processes are marked separately from external processes.
- If another system user appears on that group, Shardon kills its own backend immediately.
- The event is recorded in runtime state and JSONL events.

The first implementation is Linux and NVIDIA oriented, using provider abstractions so future alternatives can plug in cleanly.

