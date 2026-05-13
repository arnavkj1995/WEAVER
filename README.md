# WEAVER

**WEAVER** is a latent world model for long-horizon robot video prediction and evaluation.

<!-- TODO: add paper, arXiv, project website, pretrained checkpoints, and dataset links when public. -->

[![License](https://img.shields.io/badge/License-TBD-lightgrey?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?style=for-the-badge&logo=pytorch)](https://pytorch.org/)

---

## Overview

This repository contains the world-model code for **WEAVER**, a two-camera latent flow model for DROID-style robot rollouts. WEAVER predicts future latent observations conditioned on image latents, robot states, actions, language/task features, and optional sparse memory frames.

The codebase includes:

- Latent flow world model with image, state, action, and text conditioning.
- DROID-style dataloaders with precomputed latent support.
- Reward-model and critic heads for auxiliary world-model training.
- Diffusion-forcing rollouts with linear, cosine, power, and sigmoid inference schedules.
- KV-cache accelerated generation for faster evaluation.
- Offline FID, FVD, LPIPS, PSNR, and SSIM computation from saved view arrays.

The repository is intentionally scoped to the world-model stack. Policy learning, plotting artifacts, and experiment-specific analysis scripts are not part of the core package.

---

## Repository Structure

The main package lives in [`weaver/`](./weaver).

### Key paths

- **World model code:** [`weaver/wm/`](./weaver/wm/)  
  Encoders, decoders, transformer blocks, flow model, and KV-cache support.

- **Dataset loading:** [`weaver/datasets/`](./weaver/datasets/)  
  Runtime dataloaders for preprocessed DROID-style trajectories.

- **Preprocessing utilities:** [`datasets/preprocess_droid.py`](./datasets/preprocess_droid.py)  
  Offline helpers for normalization stats, video conversion, text features, and SD3 latent encoding.

- **Pretraining entry:** [`weaver/pretrain.py`](./weaver/pretrain.py)

- **Finetuning entry:** [`weaver/finetune.py`](./weaver/finetune.py)

- **Video generation entry:** [`weaver/generate_views.py`](./weaver/generate_views.py)

- **Offline metric script:** [`scripts/compute_metrics.py`](./scripts/compute_metrics.py)

- **Default config:** [`weaver/config.yaml`](./weaver/config.yaml)

---

## Installation

Create the environment with `uv`:

```bash
git clone <repo-url> WEAVER
cd WEAVER
uv venv --python 3.11
source .venv/bin/activate
uv sync
```

You can also run commands directly through `uv`:

```bash
uv run python -m weaver.generate_views --help
```

---

## Training WEAVER

WEAVER expects preprocessed DROID-style data containing annotations, actions/states, view videos or precomputed view latents, text features, rewards, and normalization statistics. The data root is configured with:

```bash
dataset.path=/path/to/preprocessed_droid
```

### Pretraining

Pretrain a WEAVER world model from scratch:

```bash
python -m weaver.pretrain \
  --config weaver/config.yaml \
  dataset.path=/path/to/preprocessed_droid \
  scratch_dir=/path/to/output/model_dir
```

Common overrides:

```bash
training.batch_size=8
training.max_steps=1000000
model.val_steps=16
model.diff_forcing=True
inference.pyramid_schedule=cosine
inference.pyramid_stagger_width=1
```

Checkpoints are written to:

```text
<scratch_dir>/logs/chkpts/checkpoint.pt
```

Slurm entrypoint:

```bash
DATASET_PATH=/path/to/preprocessed_droid \
SCRATCH_DIR=/path/to/output/model_dir \
sbatch scripts/pretrain.sh
```

### Finetuning

Finetune from a pretrained checkpoint directory:

```bash
python -m weaver.finetune \
  --config weaver/config.yaml \
  --pretrained_dir /path/to/pretrained/logs/chkpts \
  --finetune_suffix finetune \
  dataset.path=/path/to/finetune_data
```

Finetuned checkpoints are written to:

```text
<scratch_dir>/logs/chkpts_<finetune_suffix>/checkpoint.pt
```

Slurm entrypoint:

```bash
PRETRAINED_DIR=/path/to/pretrained/logs/chkpts \
DATASET_PATH=/path/to/finetune_data \
FINETUNE_SUFFIX=finetune \
sbatch scripts/finetune.sh
```

### Evaluation

Generate rollout views and videos from a checkpoint:

```bash
python -m weaver.generate_views \
  --checkpoint /path/to/logs/chkpts \
  --output-dir /path/to/eval_output \
  --split val \
  --num-samples 120 \
  --num-videos 8 \
  --start-idx 20 \
  --use-real-history \
  --use-ema \
  --overrides \
    dataset.path=/path/to/eval_dataset \
    model.val_steps=27 \
    eval_horizon=5 \
    eval_bootstrap=5 \
    inference.pyramid_stagger_width=1 \
    inference.pyramid_schedule=cosine
```

This writes view arrays and optional MP4 previews:

```text
<output-dir>/views/<traj_id>/pred_view0.npy
<output-dir>/views/<traj_id>/pred_view1.npy
<output-dir>/videos/eval_sample_<traj_id>.mp4
```

Compute offline metrics from saved views:

```bash
python scripts/compute_metrics.py \
  --gt_views_dir /path/to/shared_gt/views \
  --pred_views_dir /path/to/eval_output/views \
  --cameras view0 view1 \
  --start_frame 0 \
  --num_frames 50 \
  --fvd_sliding_window \
  --fvd_window 16 \
  --fvd_stride 8 \
  --pad_short_clips \
  --output metrics.json
```

For OOD splits with held-out trajectory IDs:

```bash
python scripts/compute_metrics.py \
  --gt_views_dir /path/to/shared_gt/views \
  --pred_views_dir /path/to/eval_output/views \
  --cameras view0 view1 \
  --start_frame 0 \
  --num_frames 50 \
  --exclude_traj_ids 60-79 \
  --fvd_sliding_window \
  --fvd_window 16 \
  --fvd_stride 8 \
  --pad_short_clips \
  --output metrics_ood.json
```

Example Slurm scripts are available under [`scripts/`](./scripts/), including generation and metric jobs for common real-history evaluation settings.

The default evaluation script covers DROID val and OOD real-history generation:

```bash
CHECKPOINT=/path/to/logs/chkpts \
DATASET=droid_val \
VAL_STEPS=27 EVAL_HORIZON=5 STAGGER_WIDTH=1 SCHEDULE=cosine \
sbatch --array=0-3 scripts/evaluate_wm.sh
```

---

## Inference Schedules

WEAVER supports several rollout schedules:

```bash
inference.pyramid_schedule=linear
inference.pyramid_schedule=cosine
inference.pyramid_schedule=power
inference.pyramid_schedule=sigmoid
```

The main inference controls are:

- `model.val_steps`: number of flow denoising steps.
- `eval_horizon`: number of future frames generated per chunk.
- `eval_bootstrap`: number of generated frames fed back before the next chunk.
- `inference.pyramid_stagger_width`: offset between frame-wise denoising schedules.

For staggered inference, the effective number of function evaluations is:

```text
NFE = val_steps + eval_horizon * pyramid_stagger_width
```

Typical settings:

```bash
# Lockstep, all future frames share the same noise level.
model.val_steps=8 eval_horizon=5 inference.pyramid_stagger_width=0

# Staggered pyramid rollout.
model.val_steps=45 eval_horizon=5 inference.pyramid_stagger_width=1
```

---

## Citation

If you use WEAVER, please cite the corresponding paper once available.

```bibtex
@article{weaver2026,
  title={WEAVER: Efficient World Models for Robot Video Prediction},
  author={TBD},
  year={2026}
}
```

---

## Acknowledgements

This codebase builds on ideas and infrastructure from latent diffusion and flow matching, Dreamer-style world models, DROID robot datasets, and the original SAILOR world-model codebase.
