from pathlib import Path

from shardon_core.config.loader import load_repository_config


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config = load_repository_config(repo_root / "config")
    print(
        "validated",
        {
            "backends": len(config.backends),
            "models": len(config.models),
            "deployments": len(config.deployments),
            "gpu_groups": len(config.gpu_groups),
        },
    )


if __name__ == "__main__":
    main()
