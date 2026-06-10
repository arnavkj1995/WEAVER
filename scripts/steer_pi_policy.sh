#!/bin/bash
# Real-robot deploy loop: interleaved PI policy execution + WEAVER best-of-N steering.
#
# At each chunk boundary WEAVER imagines NUM_SAMPLES futures, scores them with
# the reward/advantage head, and executes the best one on the robot.
#
# Common use (run from the WEAVER repo root):
#   bash scripts/steer_pi_policy.sh
#
# Override any variable on the command line:
#   TASK="pick up the cup" NUM_SAMPLES=8 bash scripts/steer_pi_policy.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# Required paths — edit these or override via environment variables
# ---------------------------------------------------------------------------
CHECKPOINT=${CHECKPOINT:-/data/yilin/world_model_ckpt/chkpts_finetune_final_v2}
OUTPUT_DIR=${OUTPUT_DIR:-/data/yilin/steer_test}
DYNAMICS_MODEL=${DYNAMICS_MODEL:-weaver/dynamics/model2_15_14.pth}

# Task instruction (required — must be set here or via environment)
TASK=${TASK:-"pick up the marker and place it in the cup"}

# ---------------------------------------------------------------------------
# Steering settings
# ---------------------------------------------------------------------------
NUM_SAMPLES=${NUM_SAMPLES:-4}           # best-of-N: PI samples imagined per chunk
OPEN_LOOP_HORIZON=${OPEN_LOOP_HORIZON:-9}
SELECTION_CRITERION=${SELECTION_CRITERION:-advantage}  # reward or advantage
MAX_STEPS=${MAX_STEPS:-700}

# ---------------------------------------------------------------------------
# Robot / PI server settings
# ---------------------------------------------------------------------------
PI_HOST=${PI_HOST:-intent-chai.lan.local.cmu.edu}
PI_PORT=${PI_PORT:-8000}

# ---------------------------------------------------------------------------
# Run options
# ---------------------------------------------------------------------------
USE_EMA=${USE_EMA:-1}
TORCH_COMPILE=${TORCH_COMPILE:-1}
USE_KV_CACHE=${USE_KV_CACHE:-1}
RELABEL_ACTION=${RELABEL_ACTION:-1}

# ---------------------------------------------------------------------------
# Build command
# ---------------------------------------------------------------------------
cmd=(
  python -m weaver.steer_pi_policy
  --checkpoint      "$CHECKPOINT"
  --output-dir      "$OUTPUT_DIR"
  --dynamics-model  "$DYNAMICS_MODEL"
  --task            "$TASK"
  --num-samples     "$NUM_SAMPLES"
  --open-loop-horizon "$OPEN_LOOP_HORIZON"
  --selection-criterion "$SELECTION_CRITERION"
  --max-steps       "$MAX_STEPS"
  --pi-host         "$PI_HOST"
  --pi-port         "$PI_PORT"
)

[[ "$USE_EMA"        == "1" ]] && cmd+=(--use-ema)
[[ "$TORCH_COMPILE"  == "1" ]] && cmd+=(--torch-compile)
[[ "$USE_KV_CACHE"   == "1" ]] && cmd+=(--use-kv-cache)
[[ "$RELABEL_ACTION" == "1" ]] && cmd+=(--relabel-action)

# ---------------------------------------------------------------------------
echo "Checkpoint:    $CHECKPOINT"
echo "Output:        $OUTPUT_DIR"
echo "Dynamics:      $DYNAMICS_MODEL"
echo "Task:          $TASK"
echo "Samples:       $NUM_SAMPLES  horizon: $OPEN_LOOP_HORIZON  criterion: $SELECTION_CRITERION"
echo "Flags:         use_ema=$USE_EMA torch_compile=$TORCH_COMPILE kv_cache=$USE_KV_CACHE relabel=$RELABEL_ACTION"
echo "Command:       ${cmd[*]}"
echo

"${cmd[@]}"
