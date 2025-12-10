#!/usr/bin/env bash
set -uo pipefail

# Batch-generate LiDAR datasets for IDs 0..3040 under Dataset/converted/usd/.
# Outputs go to ./output/<id>/ by default; override with OUTPUT_ROOT env.

OUTPUT_ROOT="${OUTPUT_ROOT:-output}"
USD_ROOT="Dataset/converted/usd"
LOG_DIR="${OUTPUT_ROOT}/logs"
TIMEOUT_SEC="${TIMEOUT_SEC:-120}"

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"

for id in $(seq 0 3040); do
  usd_path="${USD_ROOT}/${id}.usd"
  if [[ ! -f "${usd_path}" ]]; then
    echo "[Skip] Missing USD: ${usd_path}"
    continue
  fi

  log_file="${LOG_DIR}/${id}.log"
  echo "[Run] id=${id} -> ${OUTPUT_ROOT}/${id}/ (log: ${log_file})"
  start_ts=$(date +%s)

  if [[ -f "${OUTPUT_ROOT}/${id}/${id}.npy" ]]; then
    echo "[Skip] id=${id} already has output at ${OUTPUT_ROOT}/${id}/${id}.npy"
    continue
  fi

  # Force use of Isaac Sim kit python by clearing conda/virtualenv hints.
  if CONDA_PREFIX="" VIRTUAL_ENV="" PYTHONPATH="" timeout --foreground "${TIMEOUT_SEC}" \
    /scratch2/yangyu/isaac/IsaacLab/isaaclab.sh -p Generator/usd_dataset.py \
      --usd "${usd_path}" \
      --output-dir "${OUTPUT_ROOT}" \
      --frames 50 \
      --warmup 5 \
      > "${log_file}" 2>&1; then
    :
  else
    echo "[Error] id=${id} failed, see ${log_file}"
    echo "${id}" >> skipped_model
  fi

  end_ts=$(date +%s)
  duration=$((end_ts - start_ts))
  echo "[Done] id=${id} in ${duration}s"
done

echo "[Done] Completed batch 0..3040"
