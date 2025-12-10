#!/bin/bash
#SBATCH --job-name=generate_usd_parallel
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-39             
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --gpus=1                   
#SBATCH --cpus-per-task=2   
#SBATCH --mem-per-cpu=8G                  
#SBATCH --tmp=10G

set -euo pipefail

# ================= Config =================
SIF_IMAGE="/cluster/scratch/yangyu1/isaaclab_1.0.sif"

HOST_CODE_DIR="/cluster/home/yangyu1/Isaac/Generate_Grasp"
HOST_DATA_DIR="/cluster/scratch/yangyu1/datasets"
HOST_CACHE_DIR="/cluster/scratch/yangyu1/.cache_parallel/${SLURM_ARRAY_TASK_ID}"

CONTAINER_CODE="/code"
CONTAINER_DATA="/data"
CONTAINER_CACHE="/isaac_cache"

SCRIPT_IN_CONTAINER="${CONTAINER_CODE}/convert_assets.py"

# ================= CHUNK =================
TOTAL_Start=0
TOTAL_END=3040
TOTAL_ITEMS=$((TOTAL_END - TOTAL_Start + 1))

NUM_TASKS=$((SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1))

CHUNK_SIZE=$(( (TOTAL_ITEMS + NUM_TASKS - 1) / NUM_TASKS ))

MY_START=$(( TOTAL_Start + SLURM_ARRAY_TASK_ID * CHUNK_SIZE ))
MY_END=$(( MY_START + CHUNK_SIZE - 1 ))

if [ $MY_END -gt $TOTAL_END ]; then
    MY_END=$TOTAL_END
fi

if [ $MY_START -gt $TOTAL_END ]; then
    echo "Task ID $SLURM_ARRAY_TASK_ID has no work. Exiting."
    exit 0
fi

ID_LIST=$(seq -s ' ' $MY_START $MY_END)

echo "Job Array ID: ${SLURM_ARRAY_TASK_ID}"
echo "Processing range: ${MY_START} to ${MY_END}"
echo "Cache dir: ${HOST_CACHE_DIR}"

# ================= MKDIR =================
mkdir -p "${HOST_CACHE_DIR}"
mkdir -p logs

# ================= ENV =================
export OMNI_USER_DATA_DIR="${CONTAINER_CACHE}/data"
export OMNI_CACHE_DIR="${CONTAINER_CACHE}/cache"
export OMNI_LOGS_DIR="${CONTAINER_CACHE}/logs"
export OMNI_SERVER_DATA_DIR="${CONTAINER_CACHE}/server"
export EXP_PATH="${CONTAINER_CACHE}"


# ================= RUN =================
apptainer exec --nv \
    --home "${HOST_CACHE_DIR}" \
    -B "${HOST_CODE_DIR}:${CONTAINER_CODE}" \
    -B "${HOST_DATA_DIR}:${CONTAINER_DATA}" \
    -B "${HOST_CACHE_DIR}:${CONTAINER_CACHE}" \
    "${SIF_IMAGE}" \
    /isaac-sim/python.sh "${SCRIPT_IN_CONTAINER}" \
    --root "${CONTAINER_DATA}" \
    --ids ${ID_LIST}

echo "Sub-job finished at $(date)"