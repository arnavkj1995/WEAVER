#!/bin/bash
# Generate WEAVER evaluation views/videos.
#
# Common use:
#   DATASET=droid_val CHECKPOINT=/path/to/logs/chkpts sbatch --array=0-3 scripts/evaluate_wm.sh
#   DATASET=ood CHECKPOINT=/path/to/logs/chkpts sbatch --array=0-3 scripts/evaluate_wm.sh
#
# Optional run-specific knobs:
#   USE_KV_CACHE=1 VAL_STEPS=27 EVAL_HORIZON=5 STAGGER_WIDTH=1 SCHEDULE=cosine RUN_NAME=my_run sbatch --array=0-3 scripts/evaluate_wm.sh

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=2:00:00
#SBATCH --partition=long
#SBATCH --array=0-7
#SBATCH --job-name=weaver_eval
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail

# source "$(dirname "$0")/_env.sh"
source ../SAILOR-FM/.venv/bin/activate

CHECKPOINT=${CHECKPOINT:-/home/mila/a/arnav-kumar.jain/scratch/SAILOR/libero/models/ft_1M_v22_2cams_bs8_gacc1_l32_h16_d1536_v-pred_SLS0.1_DFTrue_VRLFalse_RATrue_HZ8_MF6_SP0.5/logs/chkpts}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/mila/a/arnav-kumar.jain/scratch/SAILOR/Evals/DROID/weaver}
NUM_CHUNKS=${NUM_CHUNKS:-8}
START_IDX=${START_IDX:-20}
SPLIT=${SPLIT:-val}
NUM_VIDEOS=${NUM_VIDEOS:-2}
USE_REAL_HISTORY=${USE_REAL_HISTORY:-1}
DATASET=${DATASET:-droid_val}
USE_KV_CACHE=${USE_KV_CACHE:-0}

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
CHUNK_ID=$((TASK_ID % NUM_CHUNKS))

if [[ -n "${DATASET_PATH:-}" ]]; then
  case_name=${CASE_NAME:-custom}
  num_samples=${NUM_SAMPLES:-120}
else
  case "$DATASET" in
    droid_val|droid)
      case_name=droid_val
      DATASET_PATH=/home/mila/a/arnav-kumar.jain/scratch/SAILOR/DROID/preprocessed_v2
      num_samples=256
      ;;
    ood|full_eval_ours|task_data)
      case_name=full_eval_ours
      DATASET_PATH=/home/mila/a/arnav-kumar.jain/scratch/SAILOR/DROID/world_model_full_eval_ours
      num_samples=120
      ;;
    *)
      echo "Unknown DATASET=$DATASET. Use DATASET=droid_val, DATASET=ood, or set DATASET_PATH explicitly."
      exit 1
      ;;
  esac
fi

if [[ -n "${RUN_NAME:-}" ]]; then
  output_dir="$OUTPUT_ROOT/$RUN_NAME"
else
  history_tag=$([[ "$USE_REAL_HISTORY" == "1" ]] && echo real_hist || echo anchor)
  cache_tag=$([[ "$USE_KV_CACHE" == "1" ]] && echo "_kvcache" || echo "")
  output_dir="$OUTPUT_ROOT/weaver_${history_tag}_${case_name}_start${START_IDX}${cache_tag}"
fi

overrides=(dataset.path="$DATASET_PATH")
if [[ -n "${VAL_STEPS:-}" ]]; then
  overrides+=(model.val_steps="$VAL_STEPS")
fi
if [[ -n "${EVAL_HORIZON:-}" ]]; then
  overrides+=(eval_horizon="$EVAL_HORIZON")
fi
if [[ -n "${EVAL_BOOTSTRAP:-}" ]]; then
  overrides+=(eval_bootstrap="$EVAL_BOOTSTRAP")
fi
if [[ -n "${STAGGER_WIDTH:-}" ]]; then
  overrides+=(inference.pyramid_stagger_width="$STAGGER_WIDTH")
fi
if [[ -n "${SCHEDULE:-}" ]]; then
  overrides+=(inference.pyramid_schedule="$SCHEDULE")
fi
if [[ -n "${EXTRA_OVERRIDES:-}" ]]; then
  read -r -a extra <<< "$EXTRA_OVERRIDES"
  overrides+=("${extra[@]}")
fi

cmd=(
  python -m weaver.generate_views
  --checkpoint "$CHECKPOINT"
  --output-dir "$output_dir"
  --split "$SPLIT"
  --num-samples "$num_samples"
  --num-videos "$NUM_VIDEOS"
  --start-idx "$START_IDX"
  --val-chunk "$NUM_CHUNKS"
  --val-chunk-id "$CHUNK_ID"
)
if [[ "$USE_REAL_HISTORY" == "1" ]]; then
  cmd+=(--use-real-history)
fi
cmd+=(--use-ema)
if [[ "$USE_KV_CACHE" == "1" ]]; then
  cmd+=(--use-kv-cache)
fi
cmd+=(--overrides "${overrides[@]}")

echo "Checkpoint: $CHECKPOINT"
echo "Dataset name: $DATASET"
echo "Dataset: $DATASET_PATH"
echo "Output: $output_dir"
echo "Case: $case_name chunk: $CHUNK_ID/$NUM_CHUNKS num_samples: $num_samples use_ema: true use_kv_cache: $USE_KV_CACHE"
echo "Overrides: ${overrides[*]}"

"${cmd[@]}"
