from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from shardon_core.utils.files import atomic_write_text, locked_file


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    with locked_file(path.with_suffix(".lock")):
        atomic_write_text(path, yaml.safe_dump(payload, sort_keys=False))


def delete_yaml(path: Path) -> None:
    with locked_file(path.with_suffix(".lock")):
        if path.exists():
            path.unlink()
