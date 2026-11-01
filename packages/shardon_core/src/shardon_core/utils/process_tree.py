from __future__ import annotations

import subprocess


def process_tree(root_pid: int) -> set[int]:
    """Return ``root_pid`` and all of its descendants.

    vLLM allocates GPU memory in child processes (EngineCore / workers), so
    per-PID nvidia-smi accounting must cover the launched process tree.
    Uses ``ps`` so it works on Linux and macOS; falls back to the root only.
    """
    try:
        output = subprocess.run(
            ["ps", "-A", "-o", "pid=,ppid="],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except Exception:
        return {root_pid}
    children: dict[int, list[int]] = {}
    for line in output.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            pid, ppid = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        children.setdefault(ppid, []).append(pid)
    seen = {root_pid}
    stack = [root_pid]
    while stack:
        for child in children.get(stack.pop(), []):
            if child not in seen:
                seen.add(child)
                stack.append(child)
    return seen
