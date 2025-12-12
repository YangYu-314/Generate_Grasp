#!/usr/bin/env bash
set -euo pipefail

export OMNI_FORCE_HEADLESS="1" 
export OMNI_CODE_RUNNER_NO_GUI="1" 
export DISPLAY=""

SIF_IMAGE="/cluster/scratch/yangyu1/isaaclab_1.0.sif"

HOST_CODE_DIR="/cluster/home/yangyu1/Isaac/Generate_Grasp"
HOST_DATA_DIR="/cluster/scratch/yangyu1/datasets"
HOST_CACHE_DIR="/cluster/scratch/yangyu1/.cache"
HOST_ASSETS_DIR="/cluster/scratch/yangyu1/isaacsim_assets"
HOST_KIT_CACHE_DIR="${HOST_CACHE_DIR}/kit-cache"

CONTAINER_CODE="/code"
CONTAINER_DATA="/data"
CONTAINER_CACHE="/isaac_cache"
CONTAINER_ASSETS="/isaacsim_assets"

SCRIPT_IN_CONTAINER="${CONTAINER_CODE}/Generator/usd_dataset.py"
TARGET_USD="${CONTAINER_CODE}/test/usd/507.usd"
OUTPUT_DIR="${CONTAINER_CODE}/test/lidar"
ASSETS_ROOT_PATH="${CONTAINER_ASSETS}/Assets/Isaac/5.1"

mkdir -p "${HOST_CACHE_DIR}"
mkdir -p "${HOST_CODE_DIR}/test/lidar"
mkdir -p "${HOST_KIT_CACHE_DIR}"

export OMNI_USER_DATA_DIR="${CONTAINER_CACHE}/data"
export OMNI_CACHE_DIR="${CONTAINER_CACHE}/cache"
export OMNI_LOGS_DIR="${CONTAINER_CACHE}/logs"
export OMNI_SERVER_DATA_DIR="${CONTAINER_CACHE}/server"
export EXP_PATH="${CONTAINER_CACHE}"
export OMNI_ASSETS_ROOT="${CONTAINER_CODE}/assets_root"

apptainer exec --nv \
    --home "${HOST_CACHE_DIR}" \
    --env ISAAC_ASSETS_PATH="${ASSETS_ROOT_PATH}" \
    -B "${HOST_CODE_DIR}:${CONTAINER_CODE}" \
    -B "${HOST_DATA_DIR}:${CONTAINER_DATA}" \
    -B "${HOST_CACHE_DIR}:${CONTAINER_CACHE}" \
    -B "${HOST_ASSETS_DIR}:${CONTAINER_ASSETS}:ro" \
    -B "${HOST_KIT_CACHE_DIR}:/isaac-sim/kit/cache" \
    "${SIF_IMAGE}" \
    /isaac-sim/python.sh "${SCRIPT_IN_CONTAINER}" \
    --usd "${TARGET_USD}" \
    --output-dir "${OUTPUT_DIR}" \
    --include-bottom \
    --visualize 
