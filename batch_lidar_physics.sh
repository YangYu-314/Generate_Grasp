#!/bin/bash
#SBATCH --job-name=lidar_phys
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-59
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G

set -euo pipefail

mkdir -p logs

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ---------- PATH ----------``
LIDAR_DIR="/cluster/scratch/yangyu1/lidar_output"
PY_SCRIPT="/cluster/home/yangyu1/Isaac/Generate_Grasp/read_property.py"

# ---------- ENV ----------
source ~/trimesh/bin/activate

# ---------- 分块：0-3040 均分给 60 个 task ----------
TOTAL_START=0
TOTAL_END=3040
TOTAL_ITEMS=$((TOTAL_END - TOTAL_START + 1))

NUM_TASKS=$((SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1))
CHUNK_SIZE=$(( (TOTAL_ITEMS + NUM_TASKS - 1) / NUM_TASKS ))

MY_START=$(( TOTAL_START + SLURM_ARRAY_TASK_ID * CHUNK_SIZE ))
MY_END=$(( MY_START + CHUNK_SIZE - 1 ))
if [ $MY_END -gt $TOTAL_END ]; then MY_END=$TOTAL_END; fi
if [ $MY_START -gt $TOTAL_END ]; then
  echo "Task ${SLURM_ARRAY_TASK_ID} has no work."
  exit 0
fi

echo "Task ${SLURM_ARRAY_TASK_ID}: IDs ${MY_START}..${MY_END}"

# 把这段 ID 范围作为 --ids 传给你的脚本
ID_LIST=$(seq -s ' ' ${MY_START} ${MY_END})

python3 "${PY_SCRIPT}" \
  --lidar_dir "${LIDAR_DIR}" \
  --ids ${ID_LIST}
