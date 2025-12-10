#!/usr/bin/env bash
set -euo pipefail

SIF_IMAGE="/cluster/scratch/yangyu1/isaaclab_1.0.sif"

HOST_CODE_DIR="/cluster/home/yangyu1/Isaac/Generate_Grasp"
HOST_DATA_DIR="/cluster/scratch/yangyu1/datasets"
HOST_CACHE_DIR="/cluster/scratch/yangyu1/.cache"

CONTAINER_CODE="/code"
CONTAINER_DATA="/data"
CONTAINER_CACHE="/isaac_cache"

SCRIPT_IN_CONTAINER="${CONTAINER_CODE}/convert_assets.py"

mkdir -p "${HOST_CACHE_DIR}"
export OMNI_USER_DATA_DIR="${CONTAINER_CACHE}/data"
export OMNI_CACHE_DIR="${CONTAINER_CACHE}/cache"
export OMNI_LOGS_DIR="${CONTAINER_CACHE}/logs"
export OMNI_SERVER_DATA_DIR="${CONTAINER_CACHE}/server"
export EXP_PATH="${CONTAINER_CACHE}"

apptainer exec --nv \
    --home "${HOST_CACHE_DIR}" \
    -B "${HOST_CODE_DIR}:${CONTAINER_CODE}" \
    -B "${HOST_DATA_DIR}:${CONTAINER_DATA}" \
    -B "${HOST_CACHE_DIR}:${CONTAINER_CACHE}" \
    "${SIF_IMAGE}" \
    /isaac-sim/python.sh "${SCRIPT_IN_CONTAINER}" \
    --root "${CONTAINER_DATA}" \
    --ids 0