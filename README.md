# WEAVER

**WEAVER** is a latent world model for long-horizon robot video prediction and evaluation.

<!-- TODO: add paper, arXiv, project website, pretrained checkpoints, and dataset links when public. -->

[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![Website](https://img.shields.io/badge/Website-WEAVER-black?style=for-the-badge)](https://arnavkj1995.github.io/WEAVER/)

---

We introduce WEAVER: a world model architecture that satisfies the three desiderata: (i) fidelity, (ii) consistency, and (iii) efficiency. WEAVER unlocks state-of-the-art performance across policy evaluation (ρ = 0.870 correlation with real-world success rate), policy improvement (real-world success rate improvement of 38% on top of the π0.5 robot foundation model), and test-time planning (real-world success rate improvement of 14% with a 5–10× speedup over prior WMs).

![WEAVER architecture](assets/readme/weaver-architecture.png)

## 🛠️ Setup

Create a Python 3.11 environment and install dependencies with `uv`.

```bash
git clone <repo-url> WEAVER
cd WEAVER
uv venv --python 3.11
source .venv/bin/activate
uv sync
```

For optional logging and development dependencies:

```bash
uv sync --extra logging --extra dev
```

You can also run commands directly through `uv`:

```bash
uv run python -m weaver.generate_views --help
```

## 📁 Repository Structure

This repository implements the main WEAVER components: the latent flow world model, DROID dataloaders, reward and critic heads, rollout generation, and offline evaluation utilities.

```text
WEAVER
├── assets                           # README and release assets
├── datasets                         # DROID preprocessing and SD3 latent encoding utilities
├── scripts                          # Slurm launchers and offline evaluation scripts
├── weaver                           # Core WEAVER package and training/generation entrypoints
│   ├── datasets                     # Runtime DROID-style dataloaders
│   ├── utils                        # Config, checkpointing, evaluation, and metric utilities
│   └── wm                           # Latent flow world model, encoders, decoders, and transformer blocks
├── pyproject.toml                   # Package metadata and dependencies
└── README.md
```

## 🚀 Training WEAVER

WEAVER expects preprocessed DROID-style trajectories with actions, states, language features, rewards, normalization statistics, and either view videos or precomputed SD3 latents.

Encode SD3 latents for a dataset root:

```bash
python datasets/preprocess_droid.py --data_root /path/to/preprocessed_droid
```

Pretrain from scratch:

```bash
python -m weaver.pretrain \
  --config weaver/config.yaml \
  dataset.path=/path/to/preprocessed_droid \
  scratch_dir=/path/to/output/model_dir
```

Finetune from a checkpoint:

```bash
python -m weaver.finetune \
  --config weaver/config.yaml \
  --pretrained_dir /path/to/pretrained/logs/chkpts \
  --finetune_suffix finetune \
  dataset.path=/path/to/finetune_data
```

Run ReFlow post-training to distill a multi-step teacher into a faster student rollout:

```bash
python -m weaver.reflow \
  --config weaver/config.yaml \
  --pretrained_dir /path/to/teacher/logs/chkpts \
  --pretrained_ckpt_name checkpoint.pt \
  --finetune_suffix reflow \
  dataset.path=/path/to/preprocessed_droid \
  training.max_steps=4000 \
  model.rectified_teacher_steps=50 \
  model.val_steps=4
```

Slurm launchers for training workflows are available in [`scripts/`](./scripts/).

## 🔮 Inference

### Basic Inference

Generate rollout views and videos:

```bash
python -m weaver.generate_views \
  --checkpoint /path/to/logs/chkpts \
  --output-dir /path/to/eval_output \
  --split val \
  --use-real-history \
  --overrides \
    dataset.path=/path/to/eval_dataset \
    model.val_steps=27 \
    eval_horizon=5 \
    eval_bootstrap=5 \
    inference.pyramid_stagger_width=1 \
    inference.pyramid_schedule=cosine
```

### ⏱️ Inference Schedules

The main inference controls are:

- `model.val_steps`: number of flow denoising steps.
- `eval_horizon`: number of future frames generated per chunk.
- `eval_bootstrap`: number of generated frames fed back before the next chunk.
- `inference.pyramid_schedule`: denoising schedule used during rollout (`linear`, `cosine`, `power`, `sigmoid`).
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

Slurm launchers for generation workflows are available in [`scripts/`](./scripts/).

## 📊 World Model Evaluation

Compute FID, FVD, LPIPS, PSNR, and SSIM from saved views:

```bash
python scripts/compute_metrics.py \
  --gt_views_dir /path/to/shared_gt/views \
  --pred_views_dir /path/to/eval_output/views \
  --cameras wrist exterior_1_left \
  --output metrics.json
```

Slurm launchers for evaluation workflows are available in [`scripts/`](./scripts/).

---

## 📚 Citation

If you use WEAVER, please cite the corresponding paper once available.

```bibtex
@article{weaver2026,
  title={WEAVER: Efficient World Models for Robot Video Prediction},
  author={TBD},
  year={2026}
}
```

---

## 🙏 Acknowledgements

This codebase builds on ideas and infrastructure from latent diffusion and flow matching, Dreamer-style world models, DROID robot datasets, and the original SAILOR world-model codebase.
