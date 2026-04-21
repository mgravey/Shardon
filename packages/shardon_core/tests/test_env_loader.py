import os
from pathlib import Path

from shardon_core.utils.env import load_dotenv_file


def test_load_dotenv_file_sets_variables(tmp_path: Path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("HF_TOKEN=hf_local_token\nHF_HOME=/tmp/hf-cache\n", encoding="utf-8")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)

    loaded = load_dotenv_file(env_file)

    assert loaded is True
    assert os.environ["HF_TOKEN"] == "hf_local_token"
    assert os.environ["HF_HOME"] == "/tmp/hf-cache"
