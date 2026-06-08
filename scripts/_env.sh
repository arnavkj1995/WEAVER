#!/bin/bash
# Shared environment setup for WEAVER Slurm scripts.

set -euo pipefail

module load libffi
module load OpenSSL
module load cuda/12.6.0/cudnn/9.3

WEAVER_DIR=${WEAVER_DIR:-/home/mila/a/arnav-kumar.jain/LLM/WEAVER}
VENV_DIR=${VENV_DIR:-/home/mila/a/arnav-kumar.jain/LLM/SAILOR-FM/.venv}

cd "$WEAVER_DIR"
source "$VENV_DIR/bin/activate"

export PYTHONPATH="$WEAVER_DIR:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export TORCH_HOME=${TORCH_HOME:-/home/mila/a/arnav-kumar.jain/scratch/SAILOR/.cache/torch}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/home/mila/a/arnav-kumar.jain/scratch/SAILOR/.cache}
export HF_HOME=${HF_HOME:-/home/mila/a/arnav-kumar.jain/scratch/.cache/huggingface}
