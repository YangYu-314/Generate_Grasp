#!/bin/bash
#SBATCH --job-name=convert_yaml_json
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-99
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G

set -euo pipefail

HOST_CODE_DIR="/cluster/home/yangyu1/Isaac/Generate_Grasp"
YAML_ROOT="/cluster/scratch/yangyu1/grasp_output_2/datagen_sim_data/onrobot_rg6"
CONVERTER="${HOST_CODE_DIR}/GraspDataGen/scripts/graspgen/tools/convert_yaml_to_json.py"

mkdir -p logs

source ~/trimesh/bin/activate
PYTHON_BIN="python3"

MAX_ID=3040
TOTAL_IDS=$((MAX_ID + 1))

TASK_MIN=${SLURM_ARRAY_TASK_MIN:-0}
TASK_MAX=${SLURM_ARRAY_TASK_MAX:-99}
NTASKS=$((TASK_MAX - TASK_MIN + 1))

IDX=${SLURM_ARRAY_TASK_ID}

CHUNK_SIZE=$(( (TOTAL_IDS + NTASKS - 1) / NTASKS ))
START=$(( (IDX - TASK_MIN) * CHUNK_SIZE ))
END=$(( START + CHUNK_SIZE ))
if (( START >= TOTAL_IDS )); then
  echo "Task ${IDX}: start ${START} >= total ${TOTAL_IDS}, skip."
  exit 0
fi
if (( END > TOTAL_IDS )); then END=${TOTAL_IDS}; fi

echo "Task ${IDX}: processing idx ${START}..$((END-1)) of ${TOTAL_IDS}"

MISSING_LOG="logs/missing_${SLURM_JOB_ID}_${IDX}.log"
FAIL_LOG="logs/failed_${SLURM_JOB_ID}_${IDX}.log"

for ((id=START; id<END; id++)); do
  YAML_PATH="${YAML_ROOT}/${id}.yaml"
  JSON_PATH="${YAML_ROOT}/${id}.json"

  if [[ ! -f "${YAML_PATH}" ]]; then
    echo "Missing: ${YAML_PATH}" >> "${MISSING_LOG}"
    continue
  fi

  # Skip only if JSON exists and is non-empty
  if [[ -s "${JSON_PATH}" ]]; then
    continue
  fi

  if HOME=/tmp "${PYTHON_BIN}" "${CONVERTER}" "${YAML_PATH}" "${JSON_PATH}"; then
    echo "Success: ${YAML_PATH}"
  else
    echo "Failed: ${YAML_PATH}" >> "${FAIL_LOG}"
  fi
done

echo "Done. Missing log: ${MISSING_LOG} ; Fail log: ${FAIL_LOG}"
