#!/bin/bash
# Compute FID, FVD, and LPIPS from a WEAVER evaluation output folder.
#
# Usage:
#   EVAL_DIR=/path/to/eval_output sbatch scripts/compute_eval_metrics.sh
#
# EVAL_DIR may point to the evaluation root or directly to its views/ folder.

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=6:00:00
#SBATCH --partition=long
#SBATCH --job-name=weaver_metrics
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

source "$(dirname "$0")/_env.sh"

: "${EVAL_DIR:?Set EVAL_DIR=/path/to/evaluation/output}"

if [[ -d "${EVAL_DIR}/views" ]]; then
  VIEWS_DIR="${EVAL_DIR}/views"
  OUTPUT_DIR=${OUTPUT_DIR:-"${EVAL_DIR}/metrics"}
else
  VIEWS_DIR="${EVAL_DIR}"
  OUTPUT_DIR=${OUTPUT_DIR:-"${EVAL_DIR}/../metrics"}
fi

START_FRAME=${START_FRAME:-0}
NUM_FRAMES=${NUM_FRAMES:-50}
CAMERAS=${CAMERAS:-"view0 view1"}
EXCLUDE_TRAJ_IDS=${EXCLUDE_TRAJ_IDS:-}
OUTPUT_NAME=${OUTPUT_NAME:-fid_fvd_lpips.json}

mkdir -p "$OUTPUT_DIR"
read -r -a camera_args <<< "$CAMERAS"

cmd=(
  python -m weaver.evaluation.compute_metrics
  --views_dir "$VIEWS_DIR"
  --cameras "${camera_args[@]}"
  --start_frame "$START_FRAME"
  --num_frames "$NUM_FRAMES"
  --fvd_sliding_window
  --fvd_window 16
  --fvd_stride 8
  --pad_short_clips
  --skip_psnr
  --skip_ssim
  --lpips_impl torchmetrics
  --output "$OUTPUT_DIR/$OUTPUT_NAME"
)

if [[ -n "$EXCLUDE_TRAJ_IDS" ]]; then
  read -r -a exclude_args <<< "$EXCLUDE_TRAJ_IDS"
  cmd+=(--exclude_traj_ids "${exclude_args[@]}")
fi

echo "Views: $VIEWS_DIR"
echo "Cameras: ${camera_args[*]}"
echo "Frames: [$START_FRAME:$((START_FRAME + NUM_FRAMES))]"
echo "Output: $OUTPUT_DIR/$OUTPUT_NAME"

"${cmd[@]}"
