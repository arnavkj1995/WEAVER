#!/bin/bash
# Reflow post-train WEAVER on one node with 4 H100 GPUs.

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=512GB
#SBATCH --gpus=h100:4
#SBATCH --time=24:00:00
#SBATCH --tmp=2T
#SBATCH --job-name=weaver_reflow
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

source "$(dirname "$0")/_env.sh"

PRETRAINED_DIR=${PRETRAINED_DIR:-"$SCRATCH/WEAVER/models/weaver_ft_1M_ood_16k_20260521/logs/chkpts_finetune_16k"}
PRETRAINED_CKPT_NAME=${PRETRAINED_CKPT_NAME:-checkpoint.pt}
DATASET_PATH=${DATASET_PATH:-"$SCRATCH/WEAVER/DROID/world_model_full_eval_ours"}
EXP_NAME=${EXP_NAME:-weaver_reflow_4K}
FINETUNE_SUFFIX=${FINETUNE_SUFFIX:-reflow_4K}
NUM_GPUS=4
MASTER_PORT=${MASTER_PORT:-29200}

RECTIFIED_ROLLOUT_LOSS_COEFF=${RECTIFIED_ROLLOUT_LOSS_COEFF:-0.0}
STUDENT_ROLLOUT_STEPS=${STUDENT_ROLLOUT_STEPS:-4}
STUDENT_ROLLOUT_STAGGER_WIDTH=${STUDENT_ROLLOUT_STAGGER_WIDTH:-0}
TEACHER_VAL_STEPS=${TEACHER_VAL_STEPS:-50}
TEACHER_PYRAMID_STAGGER_WIDTH=${TEACHER_PYRAMID_STAGGER_WIDTH:-0}
TEACHER_PYRAMID_SCHEDULE=${TEACHER_PYRAMID_SCHEDULE:-cosine}
VAL_STEPS=${VAL_STEPS:-4}
PYRAMID_STAGGER_WIDTH=${PYRAMID_STAGGER_WIDTH:-0}
PYRAMID_SCHEDULE=${PYRAMID_SCHEDULE:-cosine}
TRAIN_STEPS=${TRAIN_STEPS:-4000}
CHECKPOINT_FREQ=${CHECKPOINT_FREQ:-250}
CHECKPOINT_MILESTONES=${CHECKPOINT_MILESTONES:-}
EXTRA_OVERRIDES=${EXTRA_OVERRIDES:-}

echo "$(date): Job ${SLURM_JOB_ID:-local} starting (restart count: ${SLURM_RESTART_COUNT:-0})"
echo "Pretrained: ${PRETRAINED_DIR}/${PRETRAINED_CKPT_NAME}"
echo "Dataset: ${DATASET_PATH}"
echo "Exp: ${EXP_NAME}"
echo "Reflow suffix: ${FINETUNE_SUFFIX}"
echo "Num GPUs: ${NUM_GPUS}"
echo "Teacher val_steps: ${TEACHER_VAL_STEPS}"
echo "Teacher stagger width: ${TEACHER_PYRAMID_STAGGER_WIDTH}"
echo "Teacher schedule: ${TEACHER_PYRAMID_SCHEDULE}"
echo "Student rollout steps: ${STUDENT_ROLLOUT_STEPS}"
echo "Student rollout stagger width: ${STUDENT_ROLLOUT_STAGGER_WIDTH}"
echo "Eval val_steps: ${VAL_STEPS}"
echo "Eval stagger width: ${PYRAMID_STAGGER_WIDTH}"
echo "Eval schedule: ${PYRAMID_SCHEDULE}"
echo "Train steps: ${TRAIN_STEPS}"
echo "Checkpoint freq: ${CHECKPOINT_FREQ}"
echo "Checkpoint milestones: ${CHECKPOINT_MILESTONES}"
echo "Extra overrides: ${EXTRA_OVERRIDES:-none}"

torchrun --standalone --nproc_per_node="${NUM_GPUS}" --master-port="${MASTER_PORT}" \
  -m weaver.reflow \
  --mode defaults \
  --pretrained_dir "${PRETRAINED_DIR}" \
  --pretrained_ckpt_name "${PRETRAINED_CKPT_NAME}" \
  --finetune_suffix "${FINETUNE_SUFFIX}" \
  scratch_dir="$SCRATCH/WEAVER/models/${EXP_NAME}" \
  exp_name="${EXP_NAME}" \
  use_wandb=False \
  save_model=True \
  ckpt_freq="${CHECKPOINT_FREQ}" \
  checkpoint_milestones="${CHECKPOINT_MILESTONES}" \
  valid_log_freq=0 \
  video_log_freq=0 \
  training.max_steps="${TRAIN_STEPS}" \
  dataset.name=DROID \
  dataset.path="${DATASET_PATH}" \
  model.rectified_rollout_loss_coeff="${RECTIFIED_ROLLOUT_LOSS_COEFF}" \
  model.rectified_student_rollout_steps="${STUDENT_ROLLOUT_STEPS}" \
  model.rectified_student_rollout_stagger_width="${STUDENT_ROLLOUT_STAGGER_WIDTH}" \
  model.rectified_teacher_steps="${TEACHER_VAL_STEPS}" \
  model.rectified_teacher_stagger_width="${TEACHER_PYRAMID_STAGGER_WIDTH}" \
  model.rectified_teacher_schedule="${TEACHER_PYRAMID_SCHEDULE}" \
  model.val_steps="${VAL_STEPS}" \
  inference.pyramid_stagger_width="${PYRAMID_STAGGER_WIDTH}" \
  inference.pyramid_schedule="${PYRAMID_SCHEDULE}" \
  eval_one_shot_chunk=False \
  ${EXTRA_OVERRIDES}
