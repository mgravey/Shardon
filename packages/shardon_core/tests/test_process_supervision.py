from pathlib import Path

from shardon_core.backends.base import ProcessSupervisor
from shardon_core.config.schemas import BackendRuntimeConfig, DeploymentConfig


def test_process_supervisor_can_start_and_kill(tmp_path: Path) -> None:
    supervisor = ProcessSupervisor(tmp_path)
    backend = BackendRuntimeConfig(
        id="backend-test",
        backend_type="mock",
        version="1.0",
        display_name="backend-test",
        runtime_dir=str(tmp_path),
        base_url="http://127.0.0.1:9",
        launch_command=["python3", "-c", "import time; time.sleep(10)"],
    )
    deployment = DeploymentConfig(
        id="dep-test",
        model_id="model",
        backend_runtime_id="backend-test",
        gpu_group_id="group",
        api_model_name="demo",
        display_name="demo",
        tasks=["chat"],
    )
    managed = supervisor.start(backend=backend, deployment=deployment)
    assert managed.pid > 0
    assert managed.log_path.exists()
    supervisor.kill(deployment.id)
    assert deployment.id not in supervisor.processes

