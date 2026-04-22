from __future__ import annotations

import tomllib
from pathlib import Path


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists() and (parent / "apps").exists():
            return parent
    raise RuntimeError("repository root not found")


def test_router_package_declares_python_multipart_dependency() -> None:
    repo_root = _repo_root()
    router_pyproject = repo_root / "apps" / "router_api" / "pyproject.toml"
    project = tomllib.loads(router_pyproject.read_text(encoding="utf-8")).get("project", {})
    dependencies = project.get("dependencies", [])
    assert any(str(dep).startswith("python-multipart") for dep in dependencies)


def test_bootstrap_installs_all_workspace_packages() -> None:
    repo_root = _repo_root()
    bootstrap = (repo_root / "scripts" / "bootstrap.sh").read_text(encoding="utf-8")
    assert "uv sync --all-packages --group dev" in bootstrap
