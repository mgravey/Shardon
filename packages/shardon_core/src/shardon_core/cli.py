from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from shardon_core.services.container import build_container
from shardon_core.services.runtime import RuntimeOperationError
from shardon_core.utils.env import load_dotenv_file


def _repo_root() -> Path:
    raw = os.environ.get("SHARDON_REPO_ROOT")
    return Path(raw).resolve() if raw else Path.cwd()


async def _run_async(args: argparse.Namespace) -> int:
    repo_root = _repo_root()
    load_dotenv_file(repo_root / ".env")
    runtime = build_container(repo_root)
    try:
        if args.runtime_command == "status":
            runtime.refresh_gpu_observations()
            runtime.enforce_keep_free()
            await runtime.refresh_backend_health()
            print(json.dumps(runtime.status(), indent=2, default=str))
            return 0
        if args.runtime_command == "load":
            result = await runtime.load_deployment(
                deployment_id=args.deployment,
                model_name=args.model,
                gpu_group_id=args.gpu_group,
                actor="cli",
            )
            print(json.dumps(result, indent=2, default=str))
            return 0
        if args.runtime_command == "unload":
            result = await runtime.unload_deployment(args.deployment, actor="cli")
            print(json.dumps(result, indent=2, default=str))
            return 0
    except RuntimeOperationError as exc:
        print(json.dumps(exc.detail, indent=2, default=str))
        return 1
    raise ValueError(f"unsupported runtime command: {args.runtime_command}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="shardon")
    subparsers = parser.add_subparsers(dest="command", required=True)

    runtime = subparsers.add_parser("runtime", help="Runtime operator commands")
    runtime_subparsers = runtime.add_subparsers(dest="runtime_command", required=True)

    runtime_subparsers.add_parser("status", help="Show runtime status")

    load = runtime_subparsers.add_parser("load", help="Manually load a deployment")
    load.add_argument("--deployment", help="Deployment id to load")
    load.add_argument("--model", help="API model alias to resolve")
    load.add_argument("--gpu-group", help="GPU group id when resolving by model")

    unload = runtime_subparsers.add_parser("unload", help="Manually unload a deployment")
    unload.add_argument("--deployment", required=True, help="Deployment id to unload")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "runtime" and args.runtime_command == "load":
        if not args.deployment and not (args.model and args.gpu_group):
            parser.error("runtime load requires --deployment or --model plus --gpu-group")
    raise SystemExit(asyncio.run(_run_async(args)))
