#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SHARDON_REPO_ROOT="$ROOT_DIR"

echo "Starting Shardon admin and router APIs from $SHARDON_REPO_ROOT"
echo "Admin API -> http://127.0.0.1:8081"
echo "Router API -> http://127.0.0.1:8080"

uv run --package shardon-admin-api shardon-admin-api &
ADMIN_PID=$!
uv run --package shardon-router-api shardon-router-api &
ROUTER_PID=$!

trap 'kill $ADMIN_PID $ROUTER_PID' EXIT
wait

