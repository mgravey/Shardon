# Drain

A drain is a blocking runtime API action that frees a GPU group.

## Behavior

- New work stops routing to the target group immediately.
- If the group is idle, the backend is stopped at once.
- If the group is busy, active requests are allowed to finish.
- When the timeout expires, Shardon force-kills the group.
- Drain lifecycle is persisted in JSON runtime state and JSONL events.

