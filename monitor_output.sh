#!/usr/bin/env bash
set -euo pipefail

# Polls OUTPUT_ROOT for new <id>/<id>.npy and triggers read_property.py for each id once.
# Usage:
#   OUTPUT_ROOT=/output POLL_INTERVAL=30 bash monitor_output.sh

OUTPUT_ROOT="${OUTPUT_ROOT:-output}"
POLL_INTERVAL="${POLL_INTERVAL:-30}"

declare -A processed

echo "[Monitor] Watching ${OUTPUT_ROOT} every ${POLL_INTERVAL}s"

while true; do
  if [[ ! -d "${OUTPUT_ROOT}" ]]; then
    sleep "${POLL_INTERVAL}"
    continue
  fi

  for dir in "${OUTPUT_ROOT}"/*/; do
    [[ -d "${dir}" ]] || continue
    id="$(basename "${dir}")"
    lidar_file="${dir%/}/${id}.npy"
    [[ -f "${lidar_file}" ]] || continue

    if [[ -n "${processed[${id}]+x}" ]]; then
      continue
    fi

    echo "[Monitor] Processing id=${id}"
    if python read_property.py --ids "${id}" --lidar_dir "${OUTPUT_ROOT}"; then
      echo "[Monitor] Completed id=${id}"
    else
      echo "[Monitor][Error] read_property failed for id=${id}"
    fi

    processed["${id}"]=1
  done

  sleep "${POLL_INTERVAL}"
done
