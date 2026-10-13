import subprocess

import pytest
from shardon_core.config.schemas import GPUDeviceConfig
from shardon_core.gpu.provider import NvidiaSMIProvider

# Deliberately synthetic identities, never derived from a test host.
UUID_A = "GPU-00000000-0000-0000-0000-000000000001"
UUID_B = "GPU-00000000-0000-0000-0000-000000000002"
ROW_A = f"{UUID_A}, 00000000:01:00.0, 1000, 2000"
ROW_B = f"{UUID_B}, 00000000:02:00.0, 3000, 4000"


def _probe(monkeypatch, rows: list[str]) -> NvidiaSMIProvider:
    def run(command, **kwargs):
        output = "\n".join(rows) if "--query-gpu=" in command[1] else ""
        return subprocess.CompletedProcess(command, 0, stdout=output)

    monkeypatch.setattr("shardon_core.gpu.provider.subprocess.run", run)
    return NvidiaSMIProvider()


def test_uuid_mapping_survives_enumeration_changes(monkeypatch) -> None:
    devices = {
        "worker-a": GPUDeviceConfig(id="worker-a", uuid=UUID_A, cuda_visible_index=1),
        "worker-b": GPUDeviceConfig(id="worker-b", uuid=UUID_B, cuda_visible_index=0),
    }
    first = _probe(monkeypatch, [ROW_A, ROW_B]).observe(devices)
    reordered = _probe(monkeypatch, [ROW_B, ROW_A]).observe(devices)
    assert first == reordered
    assert first["worker-a"].free_memory_mb == 1000
    assert first["worker-b"].free_memory_mb == 3000
    # Removing a device must not shift another GPU into its configured identity.
    remaining = _probe(monkeypatch, [ROW_B]).observe(devices)
    assert set(remaining) == {"worker-b"}


def test_missing_uuid_never_falls_back_to_replacement_at_same_pci_address(monkeypatch) -> None:
    devices = {
        "pinned": GPUDeviceConfig(id="pinned", uuid=UUID_A, pci_bus_id="00000000:02:00.0"),
        "pci-only": GPUDeviceConfig(id="pci-only", pci_bus_id="00000000:02:00.0"),
        "index-only": GPUDeviceConfig(id="index-only", cuda_visible_index=0),
    }
    observations = _probe(monkeypatch, [ROW_B]).observe(devices)
    assert set(observations) == {"pci-only"}


@pytest.mark.parametrize(
    "error",
    [FileNotFoundError("probe unavailable"), subprocess.CalledProcessError(1, "nvidia-smi")],
)
def test_failed_probe_is_unavailable_with_diagnostic(monkeypatch, caplog, error) -> None:
    def run(*args, **kwargs):
        raise error

    monkeypatch.setattr("shardon_core.gpu.provider.subprocess.run", run)
    assert (
        NvidiaSMIProvider().observe({"worker-a": GPUDeviceConfig(id="worker-a", uuid=UUID_A)}) == {}
    )
    assert "GPU inventory probe failed" in caplog.text


def test_unreadable_device_memory_does_not_hide_other_devices(monkeypatch, caplog) -> None:
    provider = _probe(monkeypatch, [ROW_A.replace("1000", "[N/A]"), ROW_B])
    observations = provider.observe(
        {
            "worker-a": GPUDeviceConfig(id="worker-a", uuid=UUID_A),
            "worker-b": GPUDeviceConfig(id="worker-b", uuid=UUID_B),
        }
    )
    assert set(observations) == {"worker-b"}
    assert "GPU inventory row unavailable" in caplog.text
