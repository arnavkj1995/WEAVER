#!/bin/bash
# Compute metrics for WEAVER saved views.
#
# CASES:
#   0: regular generation, OOD real-history h5/vs27/sw1 cosine
#   1: KV-cache generation, OOD real-history h5/vs27/sw1 cosine

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=6:00:00
#SBATCH --partition=long
#SBATCH --array=0-1
#SBATCH --job-name=weaver_metrics
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail

source ../SAILOR-FM/.venv/bin/activate
export PYTHONPATH=/home/mila/a/arnav-kumar.jain/LLM/WEAVER:${PYTHONPATH:-}
export PYTHONDONTWRITEBYTECODE=1
export TORCH_HOME=${TORCH_HOME:-/home/mila/a/arnav-kumar.jain/scratch/SAILOR/.cache/torch}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/home/mila/a/arnav-kumar.jain/scratch/SAILOR/.cache}
export HF_HOME=${HF_HOME:-/home/mila/a/arnav-kumar.jain/scratch/.cache/huggingface}

BASE=${BASE:-/home/mila/a/arnav-kumar.jain/scratch/SAILOR/Evals/DROID/weaver}
GT_VIEWS=${GT_VIEWS:-/home/mila/a/arnav-kumar.jain/scratch/SAILOR/Evals/DROID/sailor/shared_gt_start20_full_eval_val_wrist_exterior1/views}
OUT_DIR=${OUT_DIR:-eval_debugging}
mkdir -p "$OUT_DIR"

CASES=(
  "ft_1M_v22_real_hist_full_eval_val_start20_h5_vs27_sw1_cosine|regular"
  "ft_1M_v22_kvcache_real_hist_full_eval_val_start20_h5_vs27_sw1_cosine|kvcache"
)

IFS="|" read -r folder tag <<< "${CASES[$SLURM_ARRAY_TASK_ID]}"
pred_views="$BASE/$folder/views"
prefix="$OUT_DIR/${folder}_nfe32_${tag}"

echo "Folder: $folder"
echo "Tag: $tag"
echo "GT: $GT_VIEWS"
echo "Pred: $pred_views"

python scripts/compute_metrics.py \
  --gt_views_dir "$GT_VIEWS" \
  --pred_views_dir "$pred_views" \
  --cameras view0 view1 \
  --start_frame 0 \
  --num_frames 50 \
  --exclude_traj_ids 60-79 \
  --fvd_sliding_window \
  --fvd_window 16 \
  --fvd_stride 8 \
  --pad_short_clips \
  --skip_lpips \
  --skip_psnr \
  --skip_ssim \
  --output "${prefix}_fid_fvd16_stride8_50f_oodskip60_79.json"

python scripts/compute_metrics.py \
  --gt_views_dir "$GT_VIEWS" \
  --pred_views_dir "$pred_views" \
  --cameras view0 view1 \
  --start_frame 0 \
  --num_frames 50 \
  --exclude_traj_ids 60-79 \
  --skip_fid \
  --skip_fvd \
  --lpips_impl torchmetrics \
  --output "${prefix}_lpips_psnr_ssim_torchmetrics_50f_oodskip60_79.json"
