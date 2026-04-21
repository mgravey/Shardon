from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from shardon_core.config.schemas import APIKeyRecord, AdminUserRecord
from shardon_core.logging.events import EventLogger
from shardon_core.utils.files import atomic_write_json, locked_file, read_json
from shardon_core.utils.time import utc_now, utc_now_iso


def hash_secret(secret: str, *, iterations: int = 390_000) -> str:
    salt = secrets.token_hex(16)
    derived = hashlib.pbkdf2_hmac("sha256", secret.encode("utf-8"), salt.encode("utf-8"), iterations)
    return f"pbkdf2_sha256${iterations}${salt}${derived.hex()}"


def verify_secret(secret: str, encoded: str) -> bool:
    algo, raw_iterations, salt, expected = encoded.split("$", 3)
    if algo != "pbkdf2_sha256":
        return False
    derived = hashlib.pbkdf2_hmac(
        "sha256",
        secret.encode("utf-8"),
        salt.encode("utf-8"),
        int(raw_iterations),
    )
    return hmac.compare_digest(derived.hex(), expected)


@dataclass(slots=True)
class AuthResult:
    id: str
    user_name: str
    priority: int
    permissions: list[str]


class APIKeyService:
    def __init__(self, state_root: Path, event_logger: EventLogger) -> None:
        self.path = state_root / "auth" / "api_keys.json"
        self.event_logger = event_logger

    def _read(self) -> dict[str, APIKeyRecord]:
        payload = read_json(self.path, {})
        return {key: APIKeyRecord.model_validate(value) for key, value in payload.items()}

    def _write(self, records: dict[str, APIKeyRecord]) -> None:
        atomic_write_json(self.path, {key: value.model_dump(mode="json") for key, value in records.items()})

    def list_keys(self) -> list[APIKeyRecord]:
        with locked_file(self.path.with_suffix(".lock")):
            return list(self._read().values())

    def create_key(
        self,
        *,
        key_id: str,
        user_name: str,
        priority: int,
        permissions: list[str],
        actor: str,
        metadata: dict[str, str] | None = None,
    ) -> tuple[APIKeyRecord, str]:
        secret = f"shardon_{secrets.token_urlsafe(32)}"
        record = APIKeyRecord(
            id=key_id,
            user_name=user_name,
            priority=priority,
            permissions=permissions,
            secret_hash=hash_secret(secret),
            secret_prefix=secret[:12],
            created_at=utc_now_iso(),
            metadata=metadata or {},
        )
        with locked_file(self.path.with_suffix(".lock")):
            records = self._read()
            records[key_id] = record
            self._write(records)
        self.event_logger.audit("api_key.created", actor, key_id=key_id, user_name=user_name)
        return record, secret

    def revoke_key(self, key_id: str, actor: str) -> APIKeyRecord | None:
        with locked_file(self.path.with_suffix(".lock")):
            records = self._read()
            record = records.get(key_id)
            if record is None:
                return None
            record.revoked_at = utc_now_iso()
            records[key_id] = record
            self._write(records)
        self.event_logger.audit("api_key.revoked", actor, key_id=key_id)
        return record

    def authenticate(self, raw_secret: str) -> AuthResult | None:
        with locked_file(self.path.with_suffix(".lock")):
            records = self._read()
        for record in records.values():
            if record.revoked_at is None and verify_secret(raw_secret, record.secret_hash):
                return AuthResult(
                    id=record.id,
                    user_name=record.user_name,
                    priority=record.priority,
                    permissions=record.permissions,
                )
        return None


class AdminAuthService:
    def __init__(
        self,
        *,
        admin_users: dict[str, AdminUserRecord],
        state_root: Path,
        event_logger: EventLogger,
        token_ttl_seconds: int = 43_200,
    ) -> None:
        self.admin_users = admin_users
        self.event_logger = event_logger
        secret_path = state_root / "auth" / "admin_token_secret"
        if secret_path.exists():
            self.signing_secret = secret_path.read_bytes()
        else:
            secret_path.parent.mkdir(parents=True, exist_ok=True)
            self.signing_secret = os.urandom(32)
            secret_path.write_bytes(self.signing_secret)
        self.token_ttl = token_ttl_seconds

    def authenticate(self, username: str, password: str) -> str | None:
        user = self.admin_users.get(username)
        if user is None or user.disabled:
            return None
        if not verify_secret(password, user.password_hash):
            return None
        expires_at = int((utc_now() + timedelta(seconds=self.token_ttl)).timestamp())
        body = f"{username}:{expires_at}"
        signature = hmac.new(self.signing_secret, body.encode("utf-8"), hashlib.sha256).hexdigest()
        token = base64.urlsafe_b64encode(f"{body}:{signature}".encode("utf-8")).decode("ascii")
        self.event_logger.audit("admin.login", username)
        return token

    def validate_token(self, token: str) -> str | None:
        try:
            decoded = base64.urlsafe_b64decode(token.encode("ascii")).decode("utf-8")
            username, raw_expiry, signature = decoded.split(":", 2)
        except Exception:
            return None
        body = f"{username}:{raw_expiry}"
        expected = hmac.new(self.signing_secret, body.encode("utf-8"), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected):
            return None
        if datetime.fromtimestamp(int(raw_expiry), tz=UTC) < utc_now():
            return None
        if username not in self.admin_users or self.admin_users[username].disabled:
            return None
        return username
