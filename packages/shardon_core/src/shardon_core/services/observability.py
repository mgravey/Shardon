"""Read-only activity projections and durable, content-free request accounting."""

from __future__ import annotations

import codecs
import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import RLock
from typing import Any, Literal

from pydantic import BaseModel, Field

from shardon_core.config.schemas import RepositoryConfig
from shardon_core.state.models import ActiveRequest, RuntimeStateSnapshot
from shardon_core.utils.files import append_jsonl, locked_file
from shardon_core.utils.time import utc_now


class TokenUsage(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


def extract_usage(payload: Any) -> TokenUsage:
    if not isinstance(payload, dict):
        return TokenUsage()
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        response = payload.get("response")
        return extract_usage(response) if isinstance(response, dict) else TokenUsage()

    def count(*names: str) -> int | None:
        for name in names:
            value = usage.get(name)
            if type(value) is int and value >= 0:
                return value
        return None

    incoming = count("input_tokens", "prompt_tokens")
    outgoing = count("output_tokens", "completion_tokens")
    total = count("total_tokens")
    if total is None and incoming is not None and outgoing is not None:
        total = incoming + outgoing
    return TokenUsage(input_tokens=incoming, output_tokens=outgoing, total_tokens=total)


class StreamUsage:
    """Observe bounded SSE frames without modifying or buffering the forwarded stream."""

    def __init__(self) -> None:
        self.usage = TokenUsage()
        self.outcome: Literal["failed", "cancelled"] | None = None
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._line = ""
        self._data: list[str] = []
        self._size = 0
        self._discard = False
        self._cr = False

    def feed(self, chunk: bytes) -> None:
        # Split whole chunks rather than concatenating a line one character at a time.
        for part in re.split(r"([\r\n])", self._decoder.decode(chunk)):
            if not part:
                continue
            if part == "\n" and self._cr:
                self._cr = False
                continue
            self._cr = part == "\r"
            if part in {"\r", "\n"}:
                self._consume_line()
            elif not self._discard:
                self._size += len(part)
                if self._size > 1024 * 1024:
                    self._discard = True
                    self._line = "x"
                    self._data = []
                else:
                    self._line += part
            else:
                self._line = "x"

    def _consume_line(self) -> None:
        line, self._line = self._line, ""
        if not line:
            if not self._discard and self._data:
                try:
                    payload = json.loads("\n".join(self._data))
                    if isinstance(payload, dict):
                        kind = payload.get("type")
                        if kind in {"response.failed", "error"}:
                            self.outcome = "failed"
                        elif kind == "response.cancelled":
                            self.outcome = "cancelled"
                    usage = extract_usage(payload)
                    for name, value in usage.model_dump().items():
                        if value is not None:
                            setattr(self.usage, name, value)
                except (ValueError, TypeError):
                    pass
            self._data = []
            self._size = 0
            self._discard = False
        elif not self._discard and line.startswith("data:"):
            self._data.append(line[5:].removeprefix(" "))

    def finish(self) -> TokenUsage:
        self.feed(b"")
        if self._line:
            self._consume_line()
        self._consume_line()
        return self.usage


class UsageRecord(TokenUsage):
    request_id: str
    router_instance_id: str | None = None
    interruption_reason: str | None = None
    api_key_id: str
    user_name: str
    model_name: str
    requested_model_name: str | None = None
    deployment_id: str = ""
    backend_runtime_id: str = ""
    gpu_group_id: str = ""
    task: str | None = None
    request_class: Literal["interactive", "batch"] = "interactive"
    batch_id: str | None = None
    created_at: str
    started_at: str | None = None
    finished_at: str
    outcome: Literal["completed", "failed", "cancelled"]


class UsageTotals(TokenUsage):
    requests: int = 0
    completed: int = 0
    failed: int = 0
    cancelled: int = 0
    usage_available: int = 0
    usage_coverage: float = 0

    def add(self, record: UsageRecord) -> None:
        self.requests += 1
        setattr(self, record.outcome, getattr(self, record.outcome) + 1)
        self.usage_available += int(record.total_tokens is not None)
        self.usage_coverage = self.usage_available / self.requests
        for name in ("input_tokens", "output_tokens", "total_tokens"):
            value = getattr(record, name)
            if value is not None:
                setattr(self, name, (getattr(self, name) or 0) + value)


class ModelUsage(UsageTotals):
    model_name: str


class UsageRow(UsageTotals):
    identity: str
    user_names: list[str] = Field(default_factory=list)
    models: list[ModelUsage] = Field(default_factory=list)


class UsageResponse(BaseModel):
    generated_at: str
    since: str
    until: str
    timezone: str = "UTC"
    group_by: Literal["api_key", "user"]
    model_view: Literal["actual", "requested"] = "actual"
    totals: UsageTotals
    rows: list[UsageRow]


class UsageStore:
    def __init__(self, state_root: Path) -> None:
        self.root = state_root / "usage"
        self._lock = RLock()
        self._days: dict[Path, tuple[tuple[int, int], list[UsageRecord]]] = {}
        self._aggregates: dict[tuple, UsageResponse] = {}

    def _read(self, path: Path) -> list[UsageRecord]:
        if not path.exists():
            return []
        stat = path.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
        cached = self._days.get(path)
        if cached and cached[0] == signature:
            return cached[1]
        records: dict[str, UsageRecord] = {}
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                record = UsageRecord.model_validate_json(line)
            except ValueError:
                # A crash may leave a partial final append; never invent usage.
                continue
            records.setdefault(record.request_id, record)
        result = list(records.values())
        self._days[path] = (signature, result)
        return result

    def record(
        self, request: ActiveRequest, outcome: str, usage: TokenUsage | None = None
    ) -> UsageRecord:
        finished = request.finished_at or utc_now().isoformat()
        record = UsageRecord(
            **(usage or TokenUsage()).model_dump(),
            request_id=request.id,
            router_instance_id=request.router_instance_id,
            interruption_reason=request.interruption_reason,
            api_key_id=request.api_key_id,
            user_name=request.user_name,
            model_name=request.model_name,
            requested_model_name=request.requested_model_name,
            deployment_id=request.deployment_id,
            backend_runtime_id=request.backend_runtime_id,
            gpu_group_id=request.gpu_group_id,
            task=request.task,
            request_class=request.request_class,
            batch_id=request.batch_id,
            created_at=request.created_at,
            started_at=request.started_at,
            finished_at=finished,
            outcome=outcome,
        )
        day = datetime.fromisoformat(finished).astimezone(UTC).date().isoformat()
        path = self.root / f"{day}.jsonl"
        with self._lock, locked_file(self.root / ".usage.lock"):
            records = self._read(path)
            created_day = (
                datetime.fromisoformat(request.created_at).astimezone(UTC).date().isoformat()
            )
            # A retry of finalization can cross midnight (or a process restart).
            candidates = [
                path,
                *(p for p in self.root.glob("*.jsonl") if created_day <= p.stem < day),
            ]
            for candidate in candidates:
                for item in self._read(candidate):
                    if item.request_id == record.request_id:
                        return item
            # Separate any incomplete final record left by an interrupted writer.
            if path.exists() and path.stat().st_size:
                with path.open("rb+") as handle:
                    handle.seek(-1, 2)
                    if handle.read(1) != b"\n":
                        handle.write(b"\n")
            append_jsonl(path, record.model_dump(mode="json"))
            stat = path.stat()
            self._days[path] = ((stat.st_mtime_ns, stat.st_size), [*records, record])
            # Bound writer-side cache growth across day rollovers. History stays on disk.
            self._days = {p: value for p, value in self._days.items() if p in candidates}
            self._aggregates.clear()
            return record

    def summary(
        self,
        *,
        period: Literal["today", "7d", "30d"] = "today",
        group_by: Literal["api_key", "user"] = "api_key",
        model_view: Literal["actual", "requested"] = "actual",
        model: str | None = None,
        api_key: str | None = None,
        user: str | None = None,
        now: datetime | None = None,
    ) -> UsageResponse:
        now = (now or utc_now()).astimezone(UTC)
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        start -= timedelta(days={"today": 0, "7d": 6, "30d": 29}[period])
        end = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        with self._lock:
            paths = sorted(
                p
                for p in self.root.glob("*.jsonl")
                if start.date().isoformat() <= p.stem < end.date().isoformat()
            )
            signatures = tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in paths)
            key = (
                start.isoformat(),
                end.isoformat(),
                group_by,
                model_view,
                model,
                api_key,
                user,
                signatures,
            )
            if key in self._aggregates:
                return self._aggregates[key].model_copy(update={"generated_at": now.isoformat()})
            totals = UsageTotals()
            rows: dict[str, UsageRow] = {}
            models: dict[tuple[str, str], ModelUsage] = {}
            seen: set[str] = set()
            for path in paths:
                for record in self._read(path):
                    if record.request_id in seen:
                        continue
                    seen.add(record.request_id)
                    if (
                        not start
                        <= datetime.fromisoformat(record.finished_at).astimezone(UTC)
                        < end
                    ):
                        continue
                    model_name = (
                        record.requested_model_name or record.model_name
                        if model_view == "requested"
                        else record.model_name
                    )
                    if (
                        (model and model_name != model)
                        or (api_key and record.api_key_id != api_key)
                        or (user and record.user_name != user)
                    ):
                        continue
                    identity = record.api_key_id if group_by == "api_key" else record.user_name
                    row = rows.setdefault(identity, UsageRow(identity=identity))
                    if record.user_name not in row.user_names:
                        row.user_names.append(record.user_name)
                    breakdown = models.setdefault(
                        (identity, model_name), ModelUsage(model_name=model_name)
                    )
                    for item in (totals, row, breakdown):
                        item.add(record)
            for (identity, _), value in models.items():
                rows[identity].models.append(value)
            for row in rows.values():
                row.models.sort(key=lambda m: (-(m.total_tokens or 0), m.model_name))
                row.user_names.sort()
            response = UsageResponse(
                generated_at=now.isoformat(),
                since=start.isoformat(),
                until=end.isoformat(),
                group_by=group_by,
                model_view=model_view,
                totals=totals,
                rows=sorted(rows.values(), key=lambda r: (-(r.total_tokens or 0), r.identity)),
            )
            if len(self._aggregates) >= 64:
                self._aggregates.clear()
            self._aggregates[key] = response
            # Keep only the queried window in memory; files remain untouched.
            self._days = {p: value for p, value in self._days.items() if p in paths}
            return response


class ActivityRequest(BaseModel):
    id: str
    user_name: str
    api_key_id: str
    model_name: str
    requested_model_name: str | None
    deployment_id: str
    gpu_group_id: str
    endpoint: str
    priority: int
    phase: Literal["waiting", "loading", "in_flight"]
    created_at: str
    started_at: str | None
    elapsed_seconds: float
    wait_seconds: float
    wait_reason: str | None


class ActivityDeployment(BaseModel):
    id: str
    state: str
    gpu_group_id: str | None


class ModelActivity(BaseModel):
    model_name: str
    deployments: list[ActivityDeployment] = Field(default_factory=list)
    in_flight: int = 0
    waiting: int = 0
    loading: int = 0
    oldest_wait_seconds: float = 0
    requests: list[ActivityRequest] = Field(default_factory=list)


class BatchActivity(BaseModel):
    id: str
    user_name: str
    api_key_id: str
    model_name: str
    status: str
    deployment_id: str | None
    gpu_group_id: str | None
    total_items: int
    completed_items: int
    failed_items: int
    cancelled_items: int
    current_item: int | None
    remaining_items: int


class ActivityResponse(BaseModel):
    generated_at: str
    models: list[ModelActivity]
    batches: list[BatchActivity]
    in_flight: int
    waiting: int
    loading: int


ENDPOINTS = {
    "chat": "/v1/chat/completions",
    "response": "/v1/responses",
    "completion": "/v1/completions",
    "embedding": "/v1/embeddings",
    "audio_speech": "/v1/audio/speech",
    "audio_transcription": "/v1/audio/transcriptions",
    "audio_translation": "/v1/audio/translations",
}


def activity_snapshot(
    snapshot: RuntimeStateSnapshot,
    config: RepositoryConfig,
    *,
    model: str | None = None,
    api_key: str | None = None,
    user: str | None = None,
    now: datetime | None = None,
) -> ActivityResponse:
    now = now or utc_now()
    rows: dict[str, ModelActivity] = {}
    for deployment in config.deployments.values():
        if model and deployment.api_model_name != model:
            continue
        row = rows.setdefault(
            deployment.api_model_name, ModelActivity(model_name=deployment.api_model_name)
        )
        state = snapshot.deployments.get(deployment.id)
        row.deployments.append(
            ActivityDeployment(
                id=deployment.id,
                state=state.state if state else "unloaded",
                gpu_group_id=(
                    state.selected_gpu_group_id
                    or (state.gpu_group_id if state.state == "starting" else None)
                    if state
                    else None
                ),
            )
        )
    unique = {r.id: r for r in snapshot.queued_requests if r.status == "queued"}
    unique.update(snapshot.active_requests)
    for request in unique.values():
        if request.request_class == "batch" or request.status not in {"queued", "running"}:
            continue
        if (
            (model and request.model_name != model)
            or (api_key and request.api_key_id != api_key)
            or (user and request.user_name != user)
        ):
            continue
        row = rows.setdefault(request.model_name, ModelActivity(model_name=request.model_name))
        phase = "in_flight" if request.id in snapshot.active_requests else request.phase
        phase = phase if phase in {"loading", "in_flight"} else "waiting"
        created = datetime.fromisoformat(request.created_at)
        started = datetime.fromisoformat(request.started_at) if request.started_at else None
        wait = max(0, ((started or now) - created).total_seconds())
        elapsed = max(0, (now - (started or created)).total_seconds())
        row.requests.append(
            ActivityRequest(
                id=request.id,
                user_name=request.user_name,
                api_key_id=request.api_key_id,
                model_name=request.model_name,
                requested_model_name=request.requested_model_name,
                deployment_id=request.deployment_id,
                gpu_group_id=request.gpu_group_id,
                endpoint=ENDPOINTS.get(request.task or "", request.task or "unknown"),
                priority=request.priority,
                phase=phase,
                created_at=request.created_at,
                started_at=request.started_at,
                elapsed_seconds=elapsed,
                wait_seconds=wait,
                wait_reason=request.wait_reason,
            )
        )
        if phase == "in_flight":
            row.in_flight += 1
        else:
            row.waiting += 1
            row.loading += int(phase == "loading")
            row.oldest_wait_seconds = max(row.oldest_wait_seconds, wait)
    batches = []
    for job in snapshot.batch_jobs.values():
        if (
            (model and job.model_name != model)
            or (api_key and job.api_key_id != api_key)
            or (user and job.user_name != user)
        ):
            continue
        current = job.current_item if job.status == "running" else None
        batches.append(
            BatchActivity(
                id=job.id,
                user_name=job.user_name,
                api_key_id=job.api_key_id,
                model_name=job.model_name,
                status=job.status,
                deployment_id=job.deployment_id,
                gpu_group_id=job.gpu_group_id,
                total_items=job.total_items,
                completed_items=job.completed_items,
                failed_items=job.failed_items,
                cancelled_items=job.cancelled_items,
                current_item=current,
                remaining_items=max(
                    0,
                    job.total_items
                    - job.completed_items
                    - job.failed_items
                    - job.cancelled_items
                    - int(current is not None),
                ),
            )
        )
    result = sorted(rows.values(), key=lambda r: (-(r.in_flight + r.waiting), r.model_name))
    if api_key or user:
        result = [r for r in result if r.requests]
    for row in result:
        row.requests.sort(key=lambda r: (r.phase != "in_flight", r.created_at, r.id))
    return ActivityResponse(
        generated_at=now.isoformat(),
        models=result,
        batches=sorted(batches, key=lambda b: (b.status not in {"queued", "running"}, b.id)),
        in_flight=sum(r.in_flight for r in result),
        waiting=sum(r.waiting for r in result),
        loading=sum(r.loading for r in result),
    )
