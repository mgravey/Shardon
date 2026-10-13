# Activity and usage

The dashboard header displays currently loaded model names in large type, with
GPU placement. It refreshes every two seconds independently of activity filters.
Multiple loaded models are all listed; an empty runtime shows “No model loaded,”
and models starting up are labelled separately. Failed refreshes retain the last
known names with a stale indicator. The live board follows this header. Filters use exact
model aliases, API key IDs, and user names, and apply to activity and history.
In-flight requests have been dispatched to a backend; this does not distinguish
GPU execution from a backend's internal queue. Requests loading a model are a
subset of waiting requests. A request is counted once even if older runtime state
contains it in both the queue and active-request collections.

Expand a model to see owners, key IDs, request IDs, endpoint, priority, elapsed
execution (or waiting) time, total waiting time, placement, and scheduler reason.
Batch jobs are listed separately. Their current item is one-based; remaining
items exclude completed, failed, cancelled, and the current in-flight item.
Queue clearing cancels waiting requests and optionally queued batch items without
cancelling requests already dispatched to backends.

## Admin endpoints

Both endpoints require an admin bearer token. They read persisted state and do
not run health checks, scheduling, or GPU enforcement.

- `GET /runtime/activity?model=demo-chat&api_key=my-key&user=alice`
  returns a generation timestamp, per-model deployment state and requests,
  aggregate interactive in-flight/waiting/loading counts, and batch progress.
- `GET /runtime/usage?period=7d&group_by=api_key&model=demo-chat`
  returns UTC window boundaries, totals, ranked rows, and per-model breakdowns.
  `period` is `today`, `7d`, or `30d`; `group_by` is `api_key` or `user`.
  `model_view` is `actual` (default) or `requested`. It controls the per-model
  breakdowns and the meaning of the `model` filter. For example,
  `model_view=requested&model=currently-loaded` totals requests that used this
  alias, even if they were served by different models at different times.
  The same optional `model`, `api_key`, and `user` filters are supported.

Today begins at midnight UTC. Seven- and thirty-day windows include today and
the preceding six or twenty-nine calendar days. Records are attributed by finish
time. The returned `until` boundary is exclusive (the next midnight UTC).

The leaderboard's **Model names** selector switches between **Actual model used**
and **Requested model name**. Switching changes attribution, not overall totals.
Direct requests retain their model names in either view. Older records without a
requested name fall back to their recorded actual model; the alias is never inferred.
The live board and loaded-model header always show actual model names.

## Accounting

History contains one record per admitted logical request, or per batch item.
Connection recovery retains the request ID rather than counting retries as new
requests. Requests rejected before admission, such as invalid credentials or an
unknown model, are not included. Queued timeouts count as failures; clearing the
queue and task cancellation count as cancellations. Interrupted streams may have
unknown usage when the backend never delivered a usage frame.

Input, output, and total tokens are nullable. Standard `prompt_tokens` and
`completion_tokens` fields are normalized to input and output; Responses usage
is read from `usage` or `response.usage`. A total may be derived by adding reported
input and output counts; no tokenizer estimates are made. Embeddings or audio
responses may report only some fields or no token counts. Aggregated fields sum
only known values and remain null when no records supply that field. Coverage is
the fraction of requests with a known total, including failed/cancelled requests
in the denominator. Reported zero is distinct from unavailable.

Existing Responses and Completions streams are observed incrementally without
changing forwarded bytes or requesting extra frames from the backend. Repeated
cumulative usage replaces previous values rather than being summed. Clients and
backends must already provide usage frames to account for streaming tokens; this
feature does not add streaming support to other endpoints.

## Persistence and refresh

Daily JSONL files under `state/usage/` retain identity, routing metadata, outcome,
timestamps, and token counts. They do not contain prompts, generated text, audio,
key secrets, or raw backend error messages. Writes use a cross-process file lock;
readers tolerate malformed or interrupted records. Repeated request IDs are
ignored. Queries cache parsed daily files and aggregates by file modification
signatures; caches are rebuildable and remain in memory. No database is used.
Files are kept until an operator removes them; automatic retention is not enabled.
Revoking a key does not remove its historical usage.

The UI polls activity every two seconds and usage every thirty seconds, pausing
when the page is hidden and refreshing on return. Failed refreshes preserve the
last successful result with a stale indicator. Configuration editors are not
refreshed by these polls. Historical charts, billing, quotas, exports, and
backend-internal execution metrics are outside this version.
