#!/bin/bash
#SBATCH --job-name=generate_grasp_missing
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-105
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=1
#SBATCH --tmp=20G

set -euo pipefail

# ================= MISSING ID LIST =================
MISSING_IDS=(
  201
  {472..479}
  {540..559}
  {827..839}
  {1052..1059}
  {1152..1159}
  1236
  {1421..1439}
  {1833..1840}
  2006
  2179
  2211
  2523
  {2734..2739}
  2917
  {2991..2999}
)

OBJ_ID="${MISSING_IDS[$SLURM_ARRAY_TASK_ID]}"
echo "Job Array ID: ${SLURM_ARRAY_TASK_ID}"
echo "OBJ_ID: ${OBJ_ID}"

# ================= CONFIG =================
SIF_IMAGE="/cluster/scratch/yangyu1/isaaclab_1.0.sif"

HOST_CODE_DIR="/cluster/home/yangyu1/Isaac/Generate_Grasp"
HOST_DATASET_DIR="/cluster/scratch/yangyu1/datasets"
HOST_OUTPUT_DIR="/cluster/scratch/yangyu1/grasp_output_2"
HOST_CACHE_DIR="/cluster/scratch/yangyu1/.cache_grasp_missing/${SLURM_ARRAY_TASK_ID}"

CONTAINER_CODE="/code"
CONTAINER_DATA="/data"
CONTAINER_OUTPUT="/output"
CONTAINER_CACHE="/isaac_cache"

SCRIPT_IN_CONTAINER="${CONTAINER_CODE}/GraspDataGen/scripts/graspgen/datagen.py"

# ================= MKDIR =================
mkdir -p "${HOST_OUTPUT_DIR}"
mkdir -p "${HOST_CACHE_DIR}"
mkdir -p logs

mkdir -p "${HOST_CACHE_DIR}/warp_cache"
mkdir -p "${HOST_CACHE_DIR}/xdg_cache"
mkdir -p "${HOST_CACHE_DIR}/kit-data"
mkdir -p "${HOST_CACHE_DIR}/tmp_json"

# ================= BUILD ONE-OBJECT JSON (NO PYTHON) =================
ONE_JSON_HOST="${HOST_CACHE_DIR}/tmp_json/one_${OBJ_ID}.json"
ONE_JSON_CONTAINER="${CONTAINER_CACHE}/tmp_json/one_${OBJ_ID}.json"
printf '{"%s.obj": 1.0}\n' "${OBJ_ID}" > "${ONE_JSON_HOST}"
echo "[info] one-json: ${ONE_JSON_HOST} => $(cat "${ONE_JSON_HOST}")"

# ================= CMD =================
INNER_CMD="cd ${CONTAINER_CODE}/GraspDataGen && \
export HOME=${CONTAINER_CACHE} && \
export XDG_CACHE_HOME=${CONTAINER_CACHE}/xdg_cache && \
export WARP_CACHE_ROOT=${CONTAINER_CACHE}/warp_cache && \
export OMNI_USER_DATA_DIR=${CONTAINER_CACHE}/data && \
export OMNI_CACHE_DIR=${CONTAINER_CACHE}/cache && \
export OMNI_LOGS_DIR=${CONTAINER_CACHE}/logs && \
/isaac-sim/python.sh ${SCRIPT_IN_CONTAINER} \
--gripper_config onrobot_rg6 \
--object_scales_json ${ONE_JSON_CONTAINER} \
--object_root ${CONTAINER_DATA}/converted \
--num_grasps 2000 \
--max_num_envs 2000 \
--sim_output_folder ${CONTAINER_OUTPUT}/datagen_sim_data \
--guess_output_folder ${CONTAINER_OUTPUT}/grasp_guess_data \
--output_failed_grasp_locations"

# ================= RUN =================
MAX_ATTEMPTS=3
attempt=1
SLEEP_SEC=5

while true; do
  echo "$(date) [info] OBJ_ID=${OBJ_ID} attempt ${attempt} starting..."

  if apptainer exec --nv \
      -B "${HOST_CODE_DIR}:${CONTAINER_CODE}" \
      -B "${HOST_DATASET_DIR}:${CONTAINER_DATA}" \
      -B "${HOST_OUTPUT_DIR}:${CONTAINER_OUTPUT}" \
      -B "${HOST_CACHE_DIR}:${CONTAINER_CACHE}" \
      -B "${HOST_CACHE_DIR}/kit-data:/isaac-sim/kit/data" \
      "${SIF_IMAGE}" \
      /bin/bash -c "${INNER_CMD}"; then

      echo "$(date) [info] OBJ_ID=${OBJ_ID} SUCCESS."
      break
  else
      status=$?
      echo "$(date) [warn] OBJ_ID=${OBJ_ID} FAILED with code $status"

      if (( attempt >= MAX_ATTEMPTS )); then
        echo "$(date) [error] OBJ_ID=${OBJ_ID} skipped after ${MAX_ATTEMPTS} failures."
        exit 1
      fi

      sleep "$SLEEP_SEC"
      attempt=$((attempt+1))
  fi
done

echo "Job finished at $(date)"
