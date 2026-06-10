# WEAVER, Better, Faster, Longer: An Effective World Model for Robotic Manipulation


<p class="authors">
  <a href="https://arnavkj1995.github.io/">Arnav Kumar Jain</a><sup>1,2,*</sup>,
  <a href="https://yilin-wu98.github.io/">Yilin Wu</a><sup>3,*</sup>,
  <a href="https://borsa.ca/">Jesse Farebrother</a><sup>1,4</sup>,
  <a href="https://gokul.dev/">Gokul Swamy</a><sup>3</sup>,
  <a href="https://www.cs.cmu.edu/~abajcsy/">Andrea Bajcsy</a><sup>3</sup>
</p>

<!-- TODO: add paper, arXiv, project website, pretrained checkpoints, and dataset links when public. -->

[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![Website](https://img.shields.io/badge/Website-WEAVER-black?style=for-the-badge)](https://arnavkj1995.github.io/WEAVER/)
[![Models](https://img.shields.io/badge/Models-%F0%9F%A4%97-yellow?style=for-the-badge)](https://huggingface.co/arnavkj1995/WEAVER)
[![Dataset](https://img.shields.io/badge/Dataset-%F0%9F%A4%97-yellow?style=for-the-badge)](https://huggingface.co/datasets/yilin-wu/droid_ood_data)

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

## Model Checkpoints and Datasets

Download all released checkpoints from [HuggingFace](https://huggingface.co/arnavkj1995/WEAVER):

```bash
hf download arnavkj1995/WEAVER \
  --local-dir checkpoints
```

This downloads the `WEAVER`, `WEAVER-FT`, and `WEAVER-ReFlow` folders. Each
folder contains `checkpoint.pt`, `config.yaml`, and `norm_stats_relabel.json`.

The OOD evaluation dataset used in this work is available at
[yilin-wu/droid_ood_data](https://huggingface.co/datasets/yilin-wu/droid_ood_data).

```bash
# Download the full dataset
git lfs install
git clone https://huggingface.co/datasets/yilin-wu/droid_ood_data
```

Or with the Python library:

```python
from huggingface_hub import snapshot_download

local_dir = snapshot_download(
    repo_id="yilin-wu/droid_ood_data",
    repo_type="dataset",
)
```

To download only annotations and metadata (without videos/latents):

```python
from huggingface_hub import snapshot_download

local_dir = snapshot_download(
    repo_id="yilin-wu/droid_ood_data",
    repo_type="dataset",
    allow_patterns=["annotations/**", "annotation_rewards/**", "norm_stats*.json"],
)
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

## 💾 Datasets

WEAVER expects preprocessed DROID-style trajectories with actions, states, language features, rewards, normalization statistics, and either view videos or precomputed SD3 latents.

Preprocess a raw DROID download into the format expected by WEAVER:

```bash
python datasets/preprocess_droid.py \
  --data_root /path/to/raw_droid \
  --output_root /path/to/preprocessed_droid
```

See the [dataset preprocessing guide](datasets/README.md) for the expected
folder structure, parallel preprocessing, and normalization statistics.

To collect and preprocess your own custom robot trajectories, see:
- [Collecting custom trajectories](#collecting-custom-trajectories) — logging real rollouts with `panda_log.py`
- [Preprocessing custom OOD data](datasets/README.md#preprocessing-customized-ood-data) — converting raw logs into WEAVER format with SD3 latents and CLIP text features

## 🚀 Training WEAVER

By default, normalization statistics are loaded from
`<dataset.path>/norm_stats_relabel.json`. Released checkpoints should bundle
this file and set `dataset.norm_stats_path=/path/to/model/norm_stats_relabel.json`
when training or running inference.

Pretrain from scratch:

```bash
DATASET_PATH=/path/to/preprocessed_droid \
SCRATCH_DIR=/path/to/output/model_dir \
sbatch scripts/pretrain.sh
```

Finetune from a checkpoint:

```bash
PRETRAINED_DIR=/path/to/pretrained/logs/chkpts \
DATASET_PATH=/path/to/finetune_data \
EXP_NAME=weaver_finetune \
FINETUNE_SUFFIX=finetune \
sbatch scripts/finetune.sh
```

Run ReFlow post-training to distill a multi-step teacher into a faster student rollout:

```bash
PRETRAINED_DIR=/path/to/teacher/logs/chkpts \
PRETRAINED_CKPT_NAME=checkpoint.pt \
DATASET_PATH=/path/to/preprocessed_droid \
EXP_NAME=weaver_reflow \
FINETUNE_SUFFIX=reflow \
sbatch scripts/reflow.sh
```

All training launchers use four H100 GPUs with distributed data parallelism.

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

Compute FID, FVD, and LPIPS from an evaluation output folder:

```bash
EVAL_DIR=/path/to/eval_output \
sbatch scripts/compute_eval_metrics.sh
```

The folder may be the evaluation root or its `views/` subdirectory. It must
contain the saved `gt_*.npy` and `pred_*.npy` camera views.

---

## 🤖 Policy Evaluation

Replay saved DROID trajectories through WEAVER to obtain per-frame reward estimates, which can be used to evaluate and rank robot policies without real-world rollouts.

For each trajectory, the script encodes real observation frames into WEAVER's latent space, autoregressively predicts future frames conditioned on recorded actions, scores the rollout with the learned reward head, and saves a side-by-side GT-vs-prediction comparison video with the reward curve overlaid.

**Using the launcher script** (recommended):

Edit the variables at the top of `scripts/replay_reward.sh` or override them inline:

```bash
# Override checkpoint, dataset, and trajectory IDs on the command line
CHECKPOINT=/path/to/chkpts \
DATASET_PATH=/path/to/dataset \
TRAJ_IDS="10 11 12" \
START_FRAME=0 \
bash scripts/replay_reward.sh
```

Key variables in the script:

| Variable | Default | Description |
|---|---|---|
| `CHECKPOINT` | — | Path to WEAVER checkpoint directory |
| `DATASET_PATH` | — | Path to preprocessed DROID-style dataset |
| `OUTPUT_DIR` | — | Directory for output videos and reward files |
| `TRAJ_IDS` | `""` | Space-separated trajectory IDs; leave empty to use `NUM_TRAJS` |
| `NUM_TRAJS` | `10` | Number of trajectories to process (used when `TRAJ_IDS` is empty) |
| `START_FRAME` | `0` | First frame index within each trajectory |
| `SPLIT` | `val` | Dataset split (`train` or `val`) |
| `RELABEL_ACTIONS` | `1` | Use relabeled (position-difference) actions |
| `SAVE_REWARDS` | `1` | Write per-frame rewards back into annotation JSON files |
| `SHOW_REWARD` | `0` | Overlay reward curve on comparison video |

**Direct command:**

```bash
python -m weaver.replay_traj_reward \
  --checkpoint /path/to/chkpts \
  --dataset-path /path/to/dataset \
  --output-dir /path/to/output \
  --split val \
  --traj-ids 10 11 12 \
  --start-frame 0 \
  --use-ema \
  --relabel-actions \
  --save-rewards
```

## ⚗️ Synthetic Data Generation

Generate synthetic action sequences with the base policy ($\pi_{0.5}$) policy and WEAVER using best-of-N sampling. For each source trajectory, the script samples a starting step from the trajectory. It uses the observations from that step to query the base policy for N action samples, rolls out each action sample with our world model WEAVER. It keeps the action generation for longer sequence by using the decoded observation from the WEAVER to query the base policy again for M chunks. It then scores each action sequence with the advantage value computed from the reward and critic heads for the entire sequence, and saves the best action candidate sequence as a labeled annotation + video. The resulting dataset can be used to finetune the base policy alone or combined with real data together to finetune the base policy.

**Prerequisites: start the OpenPI server on a GPU machine (e.g. A6000) before running:**

```bash
# On the remote GPU machine — run from the openpi repo root
uv run scripts/serve_policy.py --env DROID --num-samples 5
```

Then set `PI_HOST` to that machine's IP/hostname when launching the script below.

**Using the launcher script** (recommended):

```bash
# Override inline
CHECKPOINT=/path/to/chkpts \
FILTER_EPISODE_ID="task" \
DATASET_PATH=/path/to/dataset \
PI_HOST=<server-ip> \
DEBUG=0 \
bash scripts/synth_data_gen.sh
```

Key variables in the script:

| Variable | Default | Description |
|---|---|---|
| `CHECKPOINT` | — | Path to WEAVER checkpoint directory |
| `DATASET_PATH` | — | Path to preprocessed DROID-style source dataset |
| `OUTPUT_DIR` | — | Directory for generated trajectories and videos |
| `DYNAMICS_MODEL` | `weaver/dynamics/model2_15_14.pth` | Path to Ctrl-World dynamics model checkpoint |
| `NUM_TRAJECTORIES` | `5` | Number of trajectories to generate |
| `NUM_SAMPLES` | `5` | Best-of-N: PI samples imagined per chunk |
| `NUM_CHUNKS` | `4` | Imagination chunks per trajectory |
| `OPEN_LOOP_HORIZON` | `9` | Control steps executed per chunk |
| `SELECTION_CRITERION` | `advantage` | `reward` or `advantage` for best-of-N selection |
| `FILTER_EPISODE_ID` | `"stack"` | Substring filter on source episode IDs; empty = no filter |
| `PRED_ACTIONS` | `1` | Use predicted (dynamics-model) actions instead of recorded |
| `DEBUG` | `1` | Save per-sample thumbnail comparison videos |
| `PI_HOST` / `PI_PORT` | `localhost` / `8000` | OpenPI policy server address |

**Direct command:**

```bash
python -m weaver.synth_data_gen \
  --checkpoint /path/to/chkpts \
  --dataset-path /path/to/dataset \
  --output-dir /path/to/output \
  --dynamics-model weaver/dynamics/model2_15_14.pth \
  --num-trajectories 1000 \
  --num-samples 5 \
  --num-chunks 4 \
  --open-loop-horizon 9 \
  --selection-criterion advantage \
  --filter-episode-id stack \
  --use-ema \
  --torch-compile \
  --pred-actions \
  --debug
```

---

<a id="collecting-custom-trajectories"></a>

## 📹 Collecting Custom Trajectories

`third_party/openpi/examples/droid/panda_log.py` runs the policy on a real
Franka Panda robot and logs each rollout as a video + annotation JSON in the
same format expected by `preprocess_droid_ood.py`.

**Prerequisites:** start the OpenPI policy server on a GPU machine before running:

```bash
# On the remote GPU machine — run from the openpi repo root
uv run scripts/serve_policy.py --env DROID
```

**Log trajectories** (run from `third_party/openpi`):

```bash
uv run examples/droid/panda_log.py \
    --folder /path/to/your/task_folder \
    --external_camera left \
    --remote_host <server-ip>
```

Each rollout:
1. Prompts for a natural-language instruction
2. Executes the policy and records joint positions, gripper state, and all three camera views
3. Asks whether the rollout succeeded (`y` / `n`)
4. If confirmed, saves to `<folder>/videos/<id>.mp4` and `<folder>/annotations/<id>.json`
5. Optionally resets the robot and loops for the next trajectory

Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--folder` | `logs/` | Output directory for videos and annotations |
| `--external_camera` | required | External camera fed to the policy: `left` or `right` |
| `--remote_host` | `0.0.0.0` | IP address of the OpenPI policy server |
| `--remote_port` | `8000` | Port of the policy server |
| `--max_timesteps` | `1000` | Maximum steps per rollout |
| `--open_loop_horizon` | `9` | Steps executed per predicted action chunk |
| `--interface_cfg` | `charmander.yml` | Deoxys robot interface config |

The saved `<folder>/` can be passed directly as an `--input_roots` entry to
`preprocess_droid_ood.py` — set `--tasks none` since there is no task sub-directory:

```bash
python -m datasets.preprocess_droid_ood \
    --input_roots /path/to/your/task_folder \
    --output_root /path/to/weaver_format_data \
    --data_type train \
    --tasks none
```

---

## 📦 Converting Synthetic Data for Finetuning

After generating synthetic data (or to package any mix of real + synthetic data for fine-tuning), use the two-step pipeline below.

### Step 1 — Convert to DROID layout

`convert_synthetic_data_to_droid.py` reads one or more root directories and
assembles a unified DROID-format output folder.  Each `--tasks` entry is a
sub-folder name relative to the root (pass `none` to read `annotations/` and
`videos/` directly from the root, useful for real-data folders with no task
sub-directory).

```bash
# Synthetic chunks only
python third_party/openpi/examples/droid/convert_synthetic_data_to_droid.py \
    --input-dirs /path/to/synthetic_data_folder \
    --output-dir ../data/data_synthetic_droid \
    --tasks cup_task marker_task

# Mix: synthetic chunks + real data sub-folder
python third_party/openpi/examples/droid/convert_synthetic_data_to_droid.py \
    --input-dirs /path/to/synthetic_data_folder /path/to/real_data_folder \
    --output-dir ../data/data_mixed_droid_all \
    --tasks cup_task marker_task 

# Real data only (annotations/ sits directly under the root)
python third_party/openpi/examples/droid/convert_synthetic_data_to_droid.py \
    --input-dirs /path/to/real_data_folder \
    --output-dir ../data/data_real_droid \
    --tasks none
```

Output layout (one folder per trajectory, plus a shared annotations file):

```
<output-dir>/
    traj_0000/
        trajectory.h5
        recordings/MP4/camera_{0,1,2}.mp4
        metadata_0.json
    traj_0001/
        ...
    aggregated-annotations-030724.json
```

### Step 2 — Convert DROID layout to LeRobot format

`convert_synthetic_droid_data_to_lerobot.py` ingests the output of Step 1 and
writes a LeRobot dataset.  Pass `--repo_name` to set both the local dataset
name and the HuggingFace Hub repo ID.  Add `--push_to_hub` to upload directly.

```bash
python third_party/openpi/examples/droid/convert_synthetic_droid_data_to_lerobot.py \
    --data_dir ../data/data_mixed_droid_all \
    --repo_name your-hf-username/your-dataset-name
```

### Step 3 — Fine-tune π₀.₅ on the converted dataset

> **Before running:** open `third_party/openpi/src/openpi/training/config.py` and
> update the `repo_id` on line 905 to match the `--repo_name` you used in Step 2:
>
> ```python
> # third_party/openpi/src/openpi/training/config.py  (line 905)
> repo_id = "your-hf-username/your-dataset-name",  # ← change this
> ```

Then launch fine-tuning from the `third_party/openpi` directory:

```bash
cd third_party/openpi
uv run scripts/train.py pi05_droid_finetune_real_syn_adv \
    --exp-name=droid-20k-real-syn-finetune \
    --overwrite
```

---

## 🕹️ Test-time Steering with WEAVER

For each observation, the script queries the base policy ($\pi_{0.5}$) for N action samples, rolls each through WEAVER's world model, scores the imagined action chunks with the advantage computed from the learned reward and critic heads, and executes only the best-scoring action chunk on the robot. This closed-loop steering improves real-world task success without any additional training.

**Prerequisites: start the OpenPI server on a GPU machine (e.g. A6000) before running:**

```bash
# On the remote GPU machine — run from the openpi repo root
uv run scripts/serve_policy.py --env DROID --num-samples 5
```

Then set `PI_HOST` to that machine's IP/hostname when launching the script below.

**Using the launcher script** (recommended):

```bash
# Override task, number of samples, and server address inline
TASK="stack the red block on the blue block" \
NUM_SAMPLES=8 \
PI_HOST=<server-ip> \
bash scripts/steer_pi_policy.sh
```

Key variables in the script:

| Variable | Default | Description |
|---|---|---|
| `CHECKPOINT` | — | Path to WEAVER checkpoint directory |
| `OUTPUT_DIR` | — | Directory for output videos and debug grids |
| `DYNAMICS_MODEL` | `weaver/dynamics/model2_15_14.pth` | Path to Ctrl-World dynamics model checkpoint |
| `TASK` | — | Language instruction for the robot |
| `NUM_SAMPLES` | `4` | Best-of-N: PI samples imagined per chunk |
| `OPEN_LOOP_HORIZON` | `9` | Control steps executed before re-planning |
| `SELECTION_CRITERION` | `advantage` | `reward` or `advantage` for best-of-N selection |
| `MAX_STEPS` | `700` | Maximum control steps per trial |
| `USE_KV_CACHE` | `1` | Reuse prefix K/V across denoising steps for faster inference |
| `RELABEL_ACTION` | `1` | Use dynamics model to compute position-difference actions |
| `PI_HOST` / `PI_PORT` | `localhost` / `8000` | PI policy server address |

**Direct command:**

```bash
python -m weaver.steer_pi_policy \
  --checkpoint /path/to/chkpts \
  --output-dir /path/to/output \
  --dynamics-model weaver/dynamics/model2_15_14.pth \
  --task "pick up the marker and place it in the cup" \
  --num-samples 4 \
  --open-loop-horizon 9 \
  --selection-criterion advantage \
  --use-ema \
  --torch-compile \
  --use-kv-cache \
  --relabel-action
```

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

WEAVER is developed from the open-source video foundation model [Stable Video Diffusion](https://github.com/Stability-AI/generative-models) and multi-view world model [Ctrl-World](https://ctrl-world.github.io/). The VLA model used in this repository is from [openpi](https://github.com/Physical-Intelligence/openpi).
