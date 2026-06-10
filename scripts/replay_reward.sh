#!/bin/bash
# Replay saved DROID trajectories through WEAVER and overlay per-frame reward curves.
#
# Common use (run from the WEAVER repo root):
#   bash scripts/replay_reward.sh
#
# Override any variable on the command line:
#   CHECKPOINT=/my/chkpts TRAJ_IDS="10 11 12" bash scripts/replay_reward.sh

set -euo pipefail

# source scripts/_env.sh 2>/dev/null || true   # optional env activation

# ---------------------------------------------------------------------------
# Required paths — edit these or override via environment variables
# ---------------------------------------------------------------------------
CHECKPOINT=${CHECKPOINT:-/data/yilin/world_model_ckpt/chkpts_finetune_final_v2}
DATASET_PATH=${DATASET_PATH:-/data/yilin/world_model_data_our_50}
OUTPUT_DIR=${OUTPUT_DIR:-/data/yilin/reward_test}

# ---------------------------------------------------------------------------
# Trajectory selection
#   TRAJ_IDS: space-separated list of integer IDs, or empty to use NUM_TRAJS
#   START_FRAME: first frame index within each trajectory
# ---------------------------------------------------------------------------
TRAJ_IDS=${TRAJ_IDS:-"60"}        # e.g. "10 11 12" or "" to use --num-trajs
NUM_TRAJS=${NUM_TRAJS:-10}         # used only when TRAJ_IDS is empty
START_FRAME=${START_FRAME:-0}

# ---------------------------------------------------------------------------
# Run options
# ---------------------------------------------------------------------------
SPLIT=${SPLIT:-val}               # train or val
USE_EMA=${USE_EMA:-1}
RELABEL_ACTIONS=${RELABEL_ACTIONS:-1}
SAVE_REWARDS=${SAVE_REWARDS:-1}
SHOW_REWARD=${SHOW_REWARD:-0}
FPS=${FPS:-5}

# ---------------------------------------------------------------------------
# Build command
# ---------------------------------------------------------------------------
cmd=(
  python -m weaver.replay_traj_reward
  --checkpoint  "$CHECKPOINT"
  --dataset-path "$DATASET_PATH"
  --output-dir  "$OUTPUT_DIR"
  --split       "$SPLIT"
  --start-frame "$START_FRAME"
  --fps         "$FPS"
)

if [[ -n "$TRAJ_IDS" ]]; then
  # shellcheck disable=SC2086
  cmd+=(--traj-ids $TRAJ_IDS)
else
  cmd+=(--num-trajs "$NUM_TRAJS")
fi

[[ "$USE_EMA"          == "1" ]] && cmd+=(--use-ema)
[[ "$RELABEL_ACTIONS"  == "1" ]] && cmd+=(--relabel-actions)
[[ "$SAVE_REWARDS"     == "1" ]] && cmd+=(--save-rewards)
[[ "$SHOW_REWARD"      == "1" ]] && cmd+=(--show-reward)

# ---------------------------------------------------------------------------
echo "Checkpoint:   $CHECKPOINT"
echo "Dataset:      $DATASET_PATH"
echo "Output:       $OUTPUT_DIR"
echo "Traj IDs:     ${TRAJ_IDS:-"(first $NUM_TRAJS)"}"
echo "Start frame:  $START_FRAME  split: $SPLIT"
echo "Flags:        use_ema=$USE_EMA relabel=$RELABEL_ACTIONS save_rewards=$SAVE_REWARDS"
echo "Command:      ${cmd[*]}"
echo

"${cmd[@]}"
