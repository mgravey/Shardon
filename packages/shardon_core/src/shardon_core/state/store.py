from __future__ import annotations

from pathlib import Path
from typing import Any

from shardon_core.logging.events import EventLogger
from shardon_core.state.models import RuntimeStateSnapshot
from shardon_core.utils.files import atomic_write_json, locked_file, read_json


class RuntimeStateStore:
    def __init__(self, state_root: Path, event_logger: EventLogger) -> None:
        self.state_root = state_root
        self.event_logger = event_logger
        self.snapshot_path = state_root / "runtime.json"
        self.queue_path = state_root / "queue" / "interactive.json"
        self.batch_path = state_root / "batches" / "jobs.json"
        self.requests_path = state_root / "requests" / "active.json"
        self.health_path = state_root / "health" / "backends.json"

    def load(self) -> RuntimeStateSnapshot:
        payload = read_json(self.snapshot_path, {})
        return RuntimeStateSnapshot.model_validate(payload or {})

    def save(self, snapshot: RuntimeStateSnapshot) -> RuntimeStateSnapshot:
        with locked_file(self.snapshot_path.with_suffix(".lock")):
            atomic_write_json(self.snapshot_path, snapshot.model_dump(mode="json"))
            atomic_write_json(
                self.queue_path,
                [item.model_dump(mode="json") for item in snapshot.queued_requests],
            )
            atomic_write_json(
                self.batch_path,
                {key: value.model_dump(mode="json") for key, value in snapshot.batch_jobs.items()},
            )
            atomic_write_json(
                self.requests_path,
                {
                    key: value.model_dump(mode="json")
                    for key, value in snapshot.active_requests.items()
                },
            )
            atomic_write_json(self.health_path, snapshot.backend_health)
        return snapshot

    def mutate(self, callback: Any) -> RuntimeStateSnapshot:
        with locked_file(self.snapshot_path.with_suffix(".lock")):
            snapshot = RuntimeStateSnapshot.model_validate(read_json(self.snapshot_path, {}))
            result = callback(snapshot)
            next_snapshot = result if isinstance(result, RuntimeStateSnapshot) else snapshot
            atomic_write_json(self.snapshot_path, next_snapshot.model_dump(mode="json"))
            atomic_write_json(
                self.queue_path,
                [item.model_dump(mode="json") for item in next_snapshot.queued_requests],
            )
            atomic_write_json(
                self.batch_path,
                {
                    key: value.model_dump(mode="json")
                    for key, value in next_snapshot.batch_jobs.items()
                },
            )
            atomic_write_json(
                self.requests_path,
                {
                    key: value.model_dump(mode="json")
                    for key, value in next_snapshot.active_requests.items()
                },
            )
            atomic_write_json(self.health_path, next_snapshot.backend_health)
            return next_snapshot
