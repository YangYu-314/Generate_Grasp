#!/bin/bash
set -euo pipefail

# ================= 1. 配置 =================
SLURM_ARRAY_TASK_ID=0
SIF_IMAGE="/cluster/scratch/yangyu1/isaaclab_1.0.sif"

HOST_CODE_DIR="/cluster/home/yangyu1/Isaac/Generate_Grasp"
HOST_DATASET_DIR="/cluster/scratch/yangyu1/datasets"
HOST_OUTPUT_DIR="/cluster/scratch/yangyu1/grasp_output_debug"
HOST_CACHE_DIR="/cluster/scratch/yangyu1/.cache_grasp_debug"

CONTAINER_CODE="/code"
CONTAINER_DATA="/data"
CONTAINER_OUTPUT="/output"
CONTAINER_CACHE="/isaac_cache"

SCRIPT_IN_CONTAINER="${CONTAINER_CODE}/GraspDataGen/scripts/graspgen/datagen.py"

# ================= 2. 准备工作 =================
mkdir -p "${HOST_OUTPUT_DIR}"
mkdir -p "${HOST_CACHE_DIR}"

# 手动创建一些子目录，防止 Warp 报错
mkdir -p "${HOST_CACHE_DIR}/warp_cache"
mkdir -p "${HOST_CACHE_DIR}/xdg_cache"

CHUNK_ID_FORMATTED=$(printf "%03d" ${SLURM_ARRAY_TASK_ID})
CHUNK_FILENAME="chunk_${CHUNK_ID_FORMATTED}.json"
CHUNK_JSON_PATH_CONTAINER="${CONTAINER_DATA}/chunks/${CHUNK_FILENAME}"

echo "Debug Mode: Processing ${CHUNK_FILENAME}"

# ================= 3. 构造内部命令 (解决一切问题的核心) =================
# 我们用 && 连接多条命令，确保环境设置在 Python 启动前生效

# 1. 切换到项目根目录 (解决 bots/onrobot_rg6.usd 找不到的问题)
# 2. 强行修改 HOME (骗过底层 C++ 插件)
# 3. 设置 XDG_CACHE_HOME (解决 Warp 试图写 /cluster/home/.cache 的问题)
# 4. 设置 WARP_CACHE_ROOT (专门解决 Warp 的编译缓存)

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
--num_grasps 10 \
--max_num_envs 10 \
--sim_output_folder ${CONTAINER_OUTPUT}/datagen_sim_data \
--guess_output_folder ${CONTAINER_OUTPUT}/grasp_guess_data"

# ================= 4. 执行 =================
echo "Starting Debug with Full Environment Override..."

apptainer exec --nv \
    -B "${HOST_CODE_DIR}:${CONTAINER_CODE}" \
    -B "${HOST_DATASET_DIR}:${CONTAINER_DATA}" \
    -B "${HOST_OUTPUT_DIR}:${CONTAINER_OUTPUT}" \
    -B "${HOST_CACHE_DIR}:${CONTAINER_CACHE}" \
    "${SIF_IMAGE}" \
    /bin/bash -c "${INNER_CMD}"

echo "Debug finished."