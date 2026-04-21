from pathlib import Path

from shardon_core.auth.service import APIKeyService
from shardon_core.logging.events import EventLogger
from shardon_core.state.models import RuntimeStateSnapshot
from shardon_core.state.store import RuntimeStateStore


def test_api_key_create_authenticate_and_revoke(tmp_path: Path) -> None:
    service = APIKeyService(tmp_path, EventLogger(tmp_path))
    record, secret = service.create_key(
        key_id="key-1",
        user_name="alice",
        priority=250,
        permissions=["inference"],
        actor="admin",
    )
    auth = service.authenticate(secret)
    assert auth is not None
    assert auth.id == record.id
    assert auth.user_name == "alice"
    service.revoke_key("key-1", "admin")
    assert service.authenticate(secret) is None


def test_runtime_state_store_round_trip(tmp_path: Path) -> None:
    store = RuntimeStateStore(tmp_path, EventLogger(tmp_path))
    snapshot = RuntimeStateSnapshot()
    saved = store.save(snapshot)
    assert saved == snapshot
    loaded = store.load()
    assert loaded.model_dump() == snapshot.model_dump()

