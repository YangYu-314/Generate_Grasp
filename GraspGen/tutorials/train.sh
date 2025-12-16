#!/usr/bin/env bash
#SBATCH --job-name=graspgen_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=12:00:00

set -euo pipefail

# Optional: pass a YAML config to override defaults.
# Usage: sbatch train.sh /path/to/extra_config.yaml
EXTRA_CFG="${1:-}"
if [[ -n "${EXTRA_CFG}" ]]; then
  # 解析为绝对路径，避免被 Hydra 改工作目录后找不到文件
  if command -v readlink >/dev/null 2>&1; then
    EXTRA_CFG_ABS="$(readlink -f "${EXTRA_CFG}")"
  else
    EXTRA_CFG_ABS="$(cd "$(dirname "${EXTRA_CFG}")" && pwd)/$(basename "${EXTRA_CFG}")"
  fi
  export GRASPGEN_EXTRA_CONFIG="${EXTRA_CFG_ABS}"
  echo "Using extra config: ${GRASPGEN_EXTRA_CONFIG}"
fi

# Code directory (absolute path, since Slurm copies this script)
CODE_DIR="/cluster/home/yangyu1/Isaac/Generate_Grasp/GraspGen"

# Basic training hyperparameters
NEPOCH=5000
BATCH=8
NWORKER=4
PRINT_FREQ=25
SAVE_FREQ=10
EVAL_FREQ=25

# Log directory on cluster scratch
LOG_ROOT="/cluster/scratch/yangyu1/graspgen_train"
mkdir -p "${LOG_ROOT}" logs
LOG_DIR="${LOG_ROOT}/${SLURM_JOB_NAME:-graspgen}_${SLURM_JOB_ID:-manual}"

echo "Running Training job in ${LOG_DIR}"
if [ -n "${SLURM_JOB_ID:-}" ]; then
  echo "SLURM job detected: ${SLURM_JOB_ID}"
fi

source /cluster/home/yangyu1/grasp/bin/activate

cd "${CODE_DIR}"
cd "${CODE_DIR}/scripts"

python train_graspgen.py \
    train.log_dir="${LOG_DIR}" \
    train.batch_size="${BATCH}" \
    train.num_epochs="${NEPOCH}" \
    train.num_workers="${NWORKER}" \
    train.print_freq="${PRINT_FREQ}" \
    train.save_freq="${SAVE_FREQ}" \
    train.eval_freq="${EVAL_FREQ}"
