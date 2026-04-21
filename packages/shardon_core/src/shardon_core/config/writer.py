from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from shardon_core.utils.files import atomic_write_text, ensure_parent, locked_file


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    with locked_file(path.with_suffix(".lock")):
        atomic_write_text(path, yaml.safe_dump(payload, sort_keys=False))


def ensure_symlink(link_path: Path, target_path: Path) -> None:
    with locked_file(link_path.with_suffix(".lock")):
        ensure_parent(link_path)
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        relative_target = Path(
            os.path.relpath(target_path, start=link_path.parent)
        )
        link_path.symlink_to(relative_target)


def delete_yaml(path: Path) -> None:
    with locked_file(path.with_suffix(".lock")):
        if path.exists():
            path.unlink()
