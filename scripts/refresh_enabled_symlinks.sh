#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

sync_dir() {
  local available_dir="$1"
  local enabled_dir="$2"

  mkdir -p "$enabled_dir"
  find "$enabled_dir" -maxdepth 1 \( -type l -o -type f \) -name '*.yaml' -exec rm -f {} +
  for file in "$available_dir"/*.yaml; do
    [ -e "$file" ] || continue
    ln -s "../$(basename "$available_dir")/$(basename "$file")" "$enabled_dir/$(basename "$file")"
  done
}

sync_dir "config/backends-available" "config/backends-enabled"
sync_dir "config/models-available" "config/models-enabled"
sync_dir "config/deployments-available" "config/deployments-enabled"
sync_dir "config/gpu-inventory-available" "config/gpu-inventory-enabled"
sync_dir "config/gpu-groups-available" "config/gpu-groups-enabled"
sync_dir "config/auth/admins-available" "config/auth/admins-enabled"
