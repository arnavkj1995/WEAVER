#!/bin/bash
# Preprocess raw DROID into WEAVER format using a 64-task Slurm array.

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40s:1
#SBATCH --array=0-63
#SBATCH --time=6:00:00
#SBATCH --job-name=weaver_preprocess_droid
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --open-mode=append

set -euo pipefail

WEAVER_DIR=${WEAVER_DIR:-${SLURM_SUBMIT_DIR:-/home/mila/a/arnav-kumar.jain/LLM/WEAVER}}
ENV_SCRIPT="${WEAVER_DIR}/scripts/_env.sh"
if [[ ! -f "${ENV_SCRIPT}" ]]; then
  echo "WEAVER environment script not found: ${ENV_SCRIPT}" >&2
  exit 1
fi
source "${ENV_SCRIPT}"

DATA_ROOT=${DATA_ROOT:-/home/mila/a/arnav-kumar.jain/scratch/SAILOR/DROID/droid_1.0.1}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/mila/a/arnav-kumar.jain/scratch/SAILOR/DROID/preprocessed_v2}
TOTAL_CHUNKS=64
CHUNK_ID=${SLURM_ARRAY_TASK_ID:-0}
BATCH_SIZE=${BATCH_SIZE:-64}

if [[ ! -f "${DATA_ROOT}/meta/episodes.jsonl" ]]; then
  echo "DROID metadata not found: ${DATA_ROOT}/meta/episodes.jsonl" >&2
  exit 1
fi

if (( CHUNK_ID < 0 || CHUNK_ID >= TOTAL_CHUNKS )); then
  echo "Chunk ID must be in [0, $((TOTAL_CHUNKS - 1))], got ${CHUNK_ID}" >&2
  exit 1
fi

echo "$(date): Starting DROID preprocessing chunk $((CHUNK_ID + 1))/${TOTAL_CHUNKS}"
echo "Input: ${DATA_ROOT}"
echo "Output: ${OUTPUT_ROOT}"
echo "Batch size: ${BATCH_SIZE}"

python datasets/preprocess_droid.py \
  --data_root "${DATA_ROOT}" \
  --output_root "${OUTPUT_ROOT}" \
  --batch_size "${BATCH_SIZE}" \
  --chunks "${TOTAL_CHUNKS}" \
  --chunk_id "${CHUNK_ID}"

echo "$(date): Finished DROID preprocessing chunk $((CHUNK_ID + 1))/${TOTAL_CHUNKS}"
