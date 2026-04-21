#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export SHARDON_REPO_ROOT="$ROOT_DIR"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required but was not found in PATH." >&2
  exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "npm is required but was not found in PATH." >&2
  exit 1
fi

uv run --package shardon-admin-api shardon-admin-api &
ADMIN_PID=$!
uv run --package shardon-router-api shardon-router-api &
ROUTER_PID=$!
npm --workspace apps/admin_web run dev -- --host 127.0.0.1 &
WEB_PID=$!

cleanup() {
  kill "$ADMIN_PID" "$ROUTER_PID" "$WEB_PID" 2>/dev/null || true
}

trap cleanup EXIT INT TERM
wait
