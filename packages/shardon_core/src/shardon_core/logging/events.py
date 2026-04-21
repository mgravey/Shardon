from __future__ import annotations

from pathlib import Path
from typing import Any

from shardon_core.utils.files import append_jsonl, locked_file
from shardon_core.utils.time import utc_now_iso


class EventLogger:
    def __init__(self, state_root: Path) -> None:
        self.state_root = state_root
        self.events_path = state_root / "events" / "events.jsonl"
        self.audit_path = state_root / "audit" / "audit.jsonl"

    def emit(self, category: str, message: str, **data: Any) -> dict[str, Any]:
        event = {
            "timestamp": utc_now_iso(),
            "category": category,
            "message": message,
            "data": data,
        }
        with locked_file(self.events_path.with_suffix(".lock")):
            append_jsonl(self.events_path, event)
        return event

    def audit(self, action: str, actor: str, **data: Any) -> dict[str, Any]:
        entry = {
            "timestamp": utc_now_iso(),
            "action": action,
            "actor": actor,
            "data": data,
        }
        with locked_file(self.audit_path.with_suffix(".lock")):
            append_jsonl(self.audit_path, entry)
        return entry
