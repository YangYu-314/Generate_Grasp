#!/usr/bin/env bash
set -euo pipefail

# Directory containing chunked JSON files (created separately).
CHUNK_DIR="/scratch2/yangyu/workspace/Dataset/converted/chunks"
LOG="datagen_loop_new.log"
SLEEP_SEC=1
MAX_ATTEMPTS=3
GRIPPER_DIR="datagen_sim_data/onrobot_rg6"

echo "$(date) [info] loop start; chunk dir=$CHUNK_DIR" | tee -a "$LOG"

# Iterate over all chunk json files in sorted order.
for CHUNK_JSON in $(ls "$CHUNK_DIR"/chunk_*.json | sort); do
  attempt=1
  echo "$(date) [info] starting chunk $CHUNK_JSON" | tee -a "$LOG"

  # Skip chunk if all outputs already exist (chunks are sequential groups of 20 objects)
  chunk_idx=${CHUNK_JSON##*_}
  chunk_idx=${chunk_idx%.*}
  start_obj=$((10#$chunk_idx * 20))
  end_obj=$((start_obj + 19))
  missing=()
  for obj_id in $(seq "$start_obj" "$end_obj"); do
    out_file="$GRIPPER_DIR/${obj_id}.yaml"
    if [[ ! -f "$out_file" ]]; then
      missing+=("$out_file")
    fi
  done
  if [[ ${#missing[@]} -eq 0 ]]; then
    echo "$(date) [skip] chunk $CHUNK_JSON (${start_obj}-${end_obj}) already complete in $GRIPPER_DIR; no Isaac run" | tee -a "$LOG"
    continue
  fi

  while true; do
    echo "$(date) [info] chunk $CHUNK_JSON attempt $attempt starting" | tee -a "$LOG"

    CMD="/scratch2/yangyu/isaac/IsaacLab/isaaclab.sh -p scripts/graspgen/datagen.py \
      --gripper_config onrobot_rg6 \
      --object_scales_json \"$CHUNK_JSON\" \
      --object_root /scratch2/yangyu/workspace/Dataset/converted \
      --num_grasps 2000 \
      --max_num_envs 2000"

    if eval "$CMD" >>"$LOG" 2>&1; then
      echo "$(date) [info] chunk $CHUNK_JSON attempt $attempt completed successfully" | tee -a "$LOG"
      break
    fi
    status=$?
    echo "$(date) [warn] chunk $CHUNK_JSON attempt $attempt exited with code $status, retry in ${SLEEP_SEC}s" | tee -a "$LOG"
    if (( attempt >= MAX_ATTEMPTS )); then
      echo "$(date) [skip] chunk $CHUNK_JSON skipped after ${MAX_ATTEMPTS} failed attempts; revisit later" | tee -a "$LOG"
      break
    fi
    sleep "$SLEEP_SEC"
    attempt=$((attempt+1))
  done
done

echo "$(date) [info] all chunks done" | tee -a "$LOG"
