#!/bin/bash
# Pretrain WEAVER on one node with 4 H100 GPUs.
#
# Common overrides:
#   DATASET_PATH=/path/to/data SCRATCH_DIR=/path/to/model sbatch scripts/pretrain.sh
#   EXTRA_OVERRIDES='training.max_steps=1000000 model.val_steps=16' sbatch scripts/pretrain.sh

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=512GB
#SBATCH --gpus=h100:4
#SBATCH --time=240:00:00
#SBATCH --tmp=2T
#SBATCH --job-name=weaver_pretrain
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

source "$(dirname "$0")/_env.sh"

CONFIG=${CONFIG:-weaver/config.yaml}
MODE=${MODE:-defaults}
NUM_GPUS=4
MASTER_PORT=${MASTER_PORT:-29000}
EXP_NAME=${EXP_NAME:-wm_pretrain}
SCRATCH_DIR=${SCRATCH_DIR:-"$SCRATCH/WEAVER/models/$EXP_NAME"}

overrides=()
if [[ -n "${DATASET_PATH:-}" ]]; then
  overrides+=(dataset.path="$DATASET_PATH")
fi
overrides+=(scratch_dir="$SCRATCH_DIR")
overrides+=(exp_name="$EXP_NAME")
if [[ -n "${EXTRA_OVERRIDES:-}" ]]; then
  read -r -a extra <<< "$EXTRA_OVERRIDES"
  overrides+=("${extra[@]}")
fi

echo "Config: $CONFIG"
echo "Mode: $MODE"
echo "Num GPUs: $NUM_GPUS"
echo "Overrides: ${overrides[*]:-(none)}"

torchrun --standalone --nproc_per_node="$NUM_GPUS" --master-port="$MASTER_PORT" \
  -m weaver.pretrain --config "$CONFIG" --mode "$MODE" "${overrides[@]}"
