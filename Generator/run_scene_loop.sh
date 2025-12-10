#!/usr/bin/env bash
set -euo pipefail

CONFIG_DIR="${1:-Generator/configs/scene}"
OUTPUT_ROOT="${2:-outputs}"
PY_CMD="${3:-/scratch/yangyu/isaacsim/_build/linux-x86_64/release/python.sh}"

shopt -s nullglob
configs=("$CONFIG_DIR"/train*.yaml)
shopt -u nullglob

if [ ${#configs[@]} -eq 0 ]; then
  echo "[Runner][Error] No train-prefixed YAML configs found in ${CONFIG_DIR}" >&2
  exit 1
fi

for config_path in "${configs[@]}"; do
  echo "[Runner] Launching scene for ${config_path}"
  "${PY_CMD}" -m Generator.pipeline --config "${config_path}" --output-root "${OUTPUT_ROOT}"
done
