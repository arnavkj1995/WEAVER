#!/bin/bash
# Finetune WEAVER on one node with 4 H100 GPUs.

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=512GB
#SBATCH --gpus=h100:4
#SBATCH --time=12:00:00
#SBATCH --tmp=2T
#SBATCH --job-name=weaver_finetune
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

source "$(dirname "$0")/_env.sh"

PRETRAINED_DIR=${PRETRAINED_DIR:-"$SCRATCH/WEAVER/models/ft_1M_v22_2cams_bs8_gacc1_l32_h16_d1536_v-pred_SLS0.1_DFTrue_VRLFalse_RATrue_HZ8_MF6_SP0.5/logs/chkpts"}
DATASET_PATH=${DATASET_PATH:-"$SCRATCH/WEAVER/DROID/world_model_full_eval_ours"}
EXP_NAME=${EXP_NAME:-weaver_ft_16K}
FINETUNE_SUFFIX=${FINETUNE_SUFFIX:-finetune_16K}
NUM_GPUS=4
MASTER_PORT=${MASTER_PORT:-29100}
TRAIN_STEPS=${TRAIN_STEPS:-16000}

EXTRA_OVERRIDES=${EXTRA_OVERRIDES:-}

echo "$(date): Job ${SLURM_JOB_ID} starting (restart count: ${SLURM_RESTART_COUNT:-0})"
echo "Pretrained: ${PRETRAINED_DIR}"
echo "Dataset: ${DATASET_PATH}"
echo "Exp: ${EXP_NAME}"
echo "Finetune suffix: ${FINETUNE_SUFFIX}"
echo "Num GPUs: ${NUM_GPUS}"
echo "Train steps: ${TRAIN_STEPS}"
echo "Extra overrides: ${EXTRA_OVERRIDES:-none}"

torchrun --standalone --nproc_per_node="${NUM_GPUS}" --master-port="${MASTER_PORT}" \
  -m weaver.finetune \
  --mode defaults \
  --pretrained_dir "${PRETRAINED_DIR}" \
  --finetune_suffix "${FINETUNE_SUFFIX}" \
  scratch_dir="$SCRATCH/WEAVER/models/${EXP_NAME}" \
  exp_name="${EXP_NAME}" \
  use_wandb=False \
  save_model=True \
  ckpt_freq=1000 \
  valid_log_freq=5000 \
  video_log_freq=5000 \
  training.max_steps="${TRAIN_STEPS}" \
  dataset.name=DROID \
  dataset.path="${DATASET_PATH}" \
  ${EXTRA_OVERRIDES}
