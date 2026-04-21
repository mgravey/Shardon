from __future__ import annotations

import pwd
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path

from shardon_core.config.schemas import GPUDeviceConfig
from shardon_core.state.models import GPUObservation, GPUProcessInfo


class GPUProvider(ABC):
    @abstractmethod
    def observe(self, gpu_devices: dict[str, GPUDeviceConfig]) -> dict[str, GPUObservation]:
        raise NotImplementedError


class MockGPUProvider(GPUProvider):
    def __init__(self, free_memory_mb: int = 48_000, total_memory_mb: int = 49_152) -> None:
        self.free_memory_mb = free_memory_mb
        self.total_memory_mb = total_memory_mb
        self.processes: list[GPUProcessInfo] = []

    def set_processes(self, processes: list[GPUProcessInfo]) -> None:
        self.processes = processes

    def observe(self, gpu_devices: dict[str, GPUDeviceConfig]) -> dict[str, GPUObservation]:
        observations: dict[str, GPUObservation] = {}
        for gpu_id in gpu_devices:
            processes = [item for item in self.processes if item.gpu_id == gpu_id]
            observations[gpu_id] = GPUObservation(
                gpu_id=gpu_id,
                free_memory_mb=self.free_memory_mb,
                total_memory_mb=self.total_memory_mb,
                observed_processes=processes,
            )
        return observations


class NvidiaSMIProvider(GPUProvider):
    def observe(self, gpu_devices: dict[str, GPUDeviceConfig]) -> dict[str, GPUObservation]:
        if not gpu_devices:
            return {}
        try:
            gpu_output = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=uuid,pci.bus_id,memory.free,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except Exception:
            return {}
        by_uuid: dict[str, dict[str, int | str]] = {}
        for line in gpu_output.stdout.splitlines():
            uuid, pci_bus_id, free_memory, total_memory = [part.strip() for part in line.split(",")]
            by_uuid[uuid] = {
                "pci_bus_id": pci_bus_id,
                "free_memory_mb": int(free_memory),
                "total_memory_mb": int(total_memory),
            }
        processes_by_uuid: dict[str, list[GPUProcessInfo]] = {}
        try:
            proc_output = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,gpu_uuid,used_memory",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            for line in proc_output.stdout.splitlines():
                pid_text, gpu_uuid, used_memory = [part.strip() for part in line.split(",")]
                pid = int(pid_text)
                try:
                    user_name = pwd.getpwuid(Path(f"/proc/{pid}").stat().st_uid).pw_name
                except Exception:
                    user_name = subprocess.run(
                        ["ps", "-o", "user=", "-p", str(pid)],
                        capture_output=True,
                        text=True,
                    ).stdout.strip() or "unknown"
                try:
                    command = Path(f"/proc/{pid}/cmdline").read_text(encoding="utf-8").replace("\x00", " ").strip()
                except Exception:
                    command = ""
                processes_by_uuid.setdefault(gpu_uuid, []).append(
                    GPUProcessInfo(
                        pid=pid,
                        user_name=user_name,
                        gpu_id="",
                        command=command,
                        memory_mb=int(used_memory),
                    )
                )
        except Exception:
            processes_by_uuid = {}
        observations: dict[str, GPUObservation] = {}
        for gpu_id, device in gpu_devices.items():
            match = None
            if device.uuid and device.uuid in by_uuid:
                match = by_uuid[device.uuid]
                raw_processes = processes_by_uuid.get(device.uuid, [])
            elif device.pci_bus_id:
                for uuid, item in by_uuid.items():
                    if item["pci_bus_id"] == device.pci_bus_id:
                        match = item
                        raw_processes = processes_by_uuid.get(uuid, [])
                        break
                else:
                    raw_processes = []
            else:
                raw_processes = []
            if match is None:
                continue
            observations[gpu_id] = GPUObservation(
                gpu_id=gpu_id,
                free_memory_mb=int(match["free_memory_mb"]),
                total_memory_mb=int(match["total_memory_mb"]),
                observed_processes=[
                    process.model_copy(update={"gpu_id": gpu_id}) for process in raw_processes
                ],
            )
        return observations
