#!/bin/bash
#SBATCH --job-name=generate_grasp_parallel
#SBATCH --output=logs/%x_%A_%a.out 
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-152            
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=1
#SBATCH --tmp=20G

set -euo pipefail

# ================= CONFIG =================
SIF_IMAGE="/cluster/scratch/yangyu1/isaaclab_1.0.sif"

HOST_CODE_DIR="/cluster/home/yangyu1/Isaac/Generate_Grasp"
HOST_DATASET_DIR="/cluster/scratch/yangyu1/datasets"
HOST_OUTPUT_DIR="/cluster/scratch/yangyu1/grasp_output"
HOST_CACHE_DIR="/cluster/scratch/yangyu1/.cache_grasp/${SLURM_ARRAY_TASK_ID}"

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

# Calculate chunk filename based on SLURM_ARRAY_TASK_ID
CHUNK_ID_FORMATTED=$(printf "%03d" ${SLURM_ARRAY_TASK_ID})
CHUNK_FILENAME="chunk_${CHUNK_ID_FORMATTED}.json"
CHUNK_JSON_PATH_CONTAINER="${CONTAINER_DATA}/chunks/${CHUNK_FILENAME}"

echo "Job Array ID: ${SLURM_ARRAY_TASK_ID}"
echo "Processing File: ${CHUNK_FILENAME}"

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
--object_scales_json ${CHUNK_JSON_PATH_CONTAINER} \
--object_root ${CONTAINER_DATA}/converted \
--num_grasps 2000 \
--max_num_envs 2000 \
--sim_output_folder ${CONTAINER_OUTPUT}/datagen_sim_data \
--guess_output_folder ${CONTAINER_OUTPUT}/grasp_guess_data"

# ================= RUN =================
MAX_ATTEMPTS=3
attempt=1
SLEEP_SEC=5

while true; do
  echo "$(date) [info] Chunk ${CHUNK_FILENAME} attempt ${attempt} starting..."
  
  # 使用 /bin/bash -c 执行 INNER_CMD
  if apptainer exec --nv \
      -B "${HOST_CODE_DIR}:${CONTAINER_CODE}" \
      -B "${HOST_DATASET_DIR}:${CONTAINER_DATA}" \
      -B "${HOST_OUTPUT_DIR}:${CONTAINER_OUTPUT}" \
      -B "${HOST_CACHE_DIR}:${CONTAINER_CACHE}" \
      "${SIF_IMAGE}" \
      /bin/bash -c "${INNER_CMD}"; then
      
      echo "$(date) [info] Chunk ${CHUNK_FILENAME} attempt ${attempt} SUCCESS."
      break
  else
      status=$?
      echo "$(date) [warn] Chunk ${CHUNK_FILENAME} attempt ${attempt} FAILED with code $status"
      
      if (( attempt >= MAX_ATTEMPTS )); then
        echo "$(date) [error] Skipped chunk ${CHUNK_FILENAME} after ${MAX_ATTEMPTS} failures."
        exit 1
      fi
      
      sleep "$SLEEP_SEC"
      attempt=$((attempt+1))
  fi
done

echo "Job finished at $(date)"