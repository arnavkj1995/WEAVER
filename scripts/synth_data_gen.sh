#!/bin/bash
# Generate synthetic trajectories via PI policy + WEAVER imagination (best-of-N selection).
#
# Common use (run from the WEAVER repo root):
#   bash scripts/synth_data_gen.sh
#
# Override any variable on the command line:
#   CHECKPOINT=/my/chkpts FILTER_EPISODE_ID="stack" bash scripts/synth_data_gen.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# Required paths — edit these or override via environment variables
# ---------------------------------------------------------------------------
CHECKPOINT=${CHECKPOINT:-/data/yilin/world_model_ckpt/chkpts_finetune_final_v2}
DATASET_PATH=${DATASET_PATH:-/data/yilin/world_model_data_ours_v2}
OUTPUT_DIR=${OUTPUT_DIR:-/data/yilin/syn_test}
DYNAMICS_MODEL=${DYNAMICS_MODEL:-weaver/dynamics/model2_15_14.pth}

# ---------------------------------------------------------------------------
# Generation settings
# ---------------------------------------------------------------------------
NUM_TRAJECTORIES=${NUM_TRAJECTORIES:-5}
NUM_SAMPLES=${NUM_SAMPLES:-5}       # best-of-N: number of PI samples per chunk
NUM_CHUNKS=${NUM_CHUNKS:-4}         # imagination chunks per trajectory
OPEN_LOOP_HORIZON=${OPEN_LOOP_HORIZON:-9}
BATCH_SIZE=${BATCH_SIZE:-4}

# ---------------------------------------------------------------------------
# PI policy server (must be running on a GPU machine before launching)
# ---------------------------------------------------------------------------
PI_HOST=${PI_HOST:-intent-chai.lan.local.cmu.edu}
PI_PORT=${PI_PORT:-8000}

# ---------------------------------------------------------------------------
# Filtering and selection
# ---------------------------------------------------------------------------
FILTER_EPISODE_ID=${FILTER_EPISODE_ID:-"stack"}   # substring filter on episode IDs; empty = no filter
FILTER_SUCCESS=${FILTER_SUCCESS:-0}
SELECTION_CRITERION=${SELECTION_CRITERION:-advantage}  # reward or advantage

# ---------------------------------------------------------------------------
# Run options
# ---------------------------------------------------------------------------
USE_EMA=${USE_EMA:-1}
TORCH_COMPILE=${TORCH_COMPILE:-1}
PRED_ACTIONS=${PRED_ACTIONS:-1}
DEBUG=${DEBUG:-1}
FPS=${FPS:-5}

# ---------------------------------------------------------------------------
# Build command
# ---------------------------------------------------------------------------
cmd=(
  python -m weaver.synth_data_gen
  --checkpoint      "$CHECKPOINT"
  --dataset-path    "$DATASET_PATH"
  --output-dir      "$OUTPUT_DIR"
  --dynamics-model  "$DYNAMICS_MODEL"
  --num-trajectories "$NUM_TRAJECTORIES"
  --num-samples     "$NUM_SAMPLES"
  --num-chunks      "$NUM_CHUNKS"
  --open-loop-horizon "$OPEN_LOOP_HORIZON"
  --batch-size      "$BATCH_SIZE"
  --selection-criterion "$SELECTION_CRITERION"
  --fps             "$FPS"
  --pi-host         "$PI_HOST"
  --pi-port         "$PI_PORT"
)

[[ -n "$FILTER_EPISODE_ID" ]] && cmd+=(--filter-episode-id "$FILTER_EPISODE_ID")
[[ "$FILTER_SUCCESS"  == "1" ]] && cmd+=(--filter-success)
[[ "$USE_EMA"         == "1" ]] && cmd+=(--use-ema)
[[ "$TORCH_COMPILE"   == "1" ]] && cmd+=(--torch-compile)
[[ "$PRED_ACTIONS"    == "1" ]] && cmd+=(--pred-actions)
[[ "$DEBUG"           == "1" ]] && cmd+=(--debug)

# ---------------------------------------------------------------------------
echo "Checkpoint:    $CHECKPOINT"
echo "Dataset:       $DATASET_PATH"
echo "Output:        $OUTPUT_DIR"
echo "Dynamics:      $DYNAMICS_MODEL"
echo "Trajs:         $NUM_TRAJECTORIES  samples/chunk: $NUM_SAMPLES  chunks: $NUM_CHUNKS"
echo "Selection:     $SELECTION_CRITERION  filter: '${FILTER_EPISODE_ID:-none}'"
echo "Flags:         use_ema=$USE_EMA torch_compile=$TORCH_COMPILE pred_actions=$PRED_ACTIONS debug=$DEBUG"
echo "Command:       ${cmd[*]}"
echo

"${cmd[@]}"
