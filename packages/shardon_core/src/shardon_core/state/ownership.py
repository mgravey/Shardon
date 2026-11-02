from __future__ import annotations

import fcntl
import uuid
from pathlib import Path


class RouterInstance:
    """A unique router lifetime, witnessed by an OS-held lock, never a timeout/PID.

    Keep witness files after exit: deleting them could let two processes lock
    different inodes for the same identity. All routers must share this directory
    on a filesystem with working cross-process flock semantics.
    """

    def __init__(self, state_root: Path) -> None:
        self.id = uuid.uuid4().hex
        self.root = state_root / "routers"
        self.root.mkdir(parents=True, exist_ok=True)
        self._handle = (self.root / f"{self.id}.lock").open("x+")
        fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def close(self) -> None:
        self._handle.close()

    def terminated(self, instance_id: str | None) -> bool:
        if instance_id == self.id:
            return False
        return router_instance_terminated(self.root.parent, instance_id)


def router_instance_terminated(state_root: Path, instance_id: str | None) -> bool:
    """True only when the router instance's witness lock is provably released."""
    if instance_id is None:
        return False
    # Unknown/legacy owners are not evidence of termination. Also avoid
    # interpreting persisted identifiers as arbitrary filesystem paths.
    try:
        if uuid.UUID(instance_id).hex != instance_id:
            return False
        with (state_root / "routers" / f"{instance_id}.lock").open("r+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
    except (OSError, ValueError):
        return False
