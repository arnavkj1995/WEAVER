#!/bin/bash
# Shared environment setup for WEAVER Slurm scripts.

set -euo pipefail

module load libffi
module load OpenSSL
module load cuda/12.6.0/cudnn/9.3

WEAVER_DIR=${WEAVER_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
: "${SCRATCH:?SCRATCH must point to your scratch directory}"
VENV_DIR=${VENV_DIR:-"$WEAVER_DIR/.venv"}

cd "$WEAVER_DIR"
if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
  echo "WEAVER virtual environment not found at $VENV_DIR." >&2
  echo "Create it with: uv venv --python 3.11 && uv sync" >&2
  return 1 2>/dev/null || exit 1
fi
source "$VENV_DIR/bin/activate"

export PYTHONPATH="$WEAVER_DIR:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export TORCH_HOME=${TORCH_HOME:-"$SCRATCH/WEAVER/.cache/torch"}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-"$SCRATCH/WEAVER/.cache"}
export HF_HOME=${HF_HOME:-"$SCRATCH/.cache/huggingface"}
