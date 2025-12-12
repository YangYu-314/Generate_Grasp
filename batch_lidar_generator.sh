#!/bin/bash
#SBATCH --job-name=gen_lidar
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-18
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --tmp=20G

set -euo pipefail

# ================= Config =================
SIF_IMAGE="/cluster/scratch/yangyu1/isaaclab_1.0.sif"

HOST_CODE_DIR="/cluster/home/yangyu1/Isaac/Generate_Grasp"
HOST_DATA_DIR="/cluster/scratch/yangyu1/datasets/converted"
HOST_ASSETS_DIR="/cluster/scratch/yangyu1/isaacsim_assets"
HOST_OUTPUT_DIR="/cluster/scratch/yangyu1/lidar_output"

HOST_CACHE_DIR="/cluster/scratch/yangyu1/.cache_parallel_lidar/${SLURM_ARRAY_TASK_ID}"
HOST_KIT_CACHE_DIR="${HOST_CACHE_DIR}/kit-cache"

CONTAINER_CODE="/code"
CONTAINER_DATA="/data"
CONTAINER_CACHE="/isaac_cache"
CONTAINER_ASSETS="/isaacsim_assets"
CONTAINER_OUTPUT="/output"

SCRIPT_IN_CONTAINER="${CONTAINER_CODE}/Generator/usd_dataset.py"
INPUT_ROOT_IN_CONTAINER="${CONTAINER_DATA}"

ASSETS_ROOT_PATH="${CONTAINER_ASSETS}/Assets/Isaac/5.1"

# ================= CHUNK CALCULATION =================
# TOTAL_Start=0
# TOTAL_END=3040
TOTAL_Start=2575
TOTAL_END=2612
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
echo "Output Directory: ${HOST_OUTPUT_DIR}"

# ================= PREPARE DIRECTORIES =================
mkdir -p "${HOST_CACHE_DIR}"
mkdir -p "${HOST_KIT_CACHE_DIR}"
mkdir -p "${HOST_OUTPUT_DIR}"
mkdir -p logs

# ================= ENV VARIABLES =================

export OMNI_FORCE_HEADLESS="1"
export OMNI_CODE_RUNNER_NO_GUI="1"
export DISPLAY=""
export OMNI_USER_DATA_DIR="${CONTAINER_CACHE}/data"
export OMNI_CACHE_DIR="${CONTAINER_CACHE}/cache"
export OMNI_LOGS_DIR="${CONTAINER_CACHE}/logs"
export OMNI_SERVER_DATA_DIR="${CONTAINER_CACHE}/server"
export EXP_PATH="${CONTAINER_CACHE}"
export OMNI_ASSETS_ROOT="${CONTAINER_CODE}/assets_root"

# ================= RUN =================

# 修改说明：使用 Shell 循环逐个 ID 启动 Python。
# 这样每次处理完一个文件，Python 进程完全退出，强制回收所有显存和 CUDA 状态。
# 只有这样才能避免 Isaac Sim RTX Renderer 的内存泄漏导致的 CUDA Error。

for CURRENT_ID in $ID_LIST; do
    echo "----------------------------------------------------------------"
    echo "Starting Process for Object ID: ${CURRENT_ID} at $(date)"

    # 注意：这里加了 '|| echo ...'
    # 因为 set -e 开启了，如果 apptainer 返回错误（比如某个文件坏了），脚本会直接退出。
    # 加上这个逻辑可以保证即使一个 ID 失败，也会继续跑下一个 ID。
    
    apptainer exec --nv \
        --home "${HOST_CACHE_DIR}" \
        --env ISAAC_ASSETS_PATH="${ASSETS_ROOT_PATH}" \
        -B "${HOST_CODE_DIR}:${CONTAINER_CODE}" \
        -B "${HOST_DATA_DIR}:${CONTAINER_DATA}" \
        -B "${HOST_CACHE_DIR}:${CONTAINER_CACHE}" \
        -B "${HOST_OUTPUT_DIR}:${CONTAINER_OUTPUT}" \
        -B "${HOST_ASSETS_DIR}:${CONTAINER_ASSETS}:ro" \
        -B "${HOST_KIT_CACHE_DIR}:/isaac-sim/kit/cache" \
        "${SIF_IMAGE}" \
        /isaac-sim/python.sh "${SCRIPT_IN_CONTAINER}" \
        --root "${INPUT_ROOT_IN_CONTAINER}" \
        --output-dir "${CONTAINER_OUTPUT}" \
        --ids "${CURRENT_ID}" \
        --include-bottom \
        --frames 50 \
        --warmup 10 \
        --visualize || echo "[WARNING] Failed to process ID: ${CURRENT_ID}"

done

echo "Job ${SLURM_ARRAY_TASK_ID} finished at $(date)"