from __future__ import annotations

from pathlib import Path

from shardon_core.gpu.provider import GPUProvider
from shardon_core.services.runtime import ShardonRuntime


def build_container(repo_root: Path, gpu_provider: GPUProvider | None = None) -> ShardonRuntime:
    return ShardonRuntime(repo_root=repo_root, gpu_provider=gpu_provider)
