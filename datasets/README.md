# Preprocessing

Download the raw [DROID dataset](https://huggingface.co/datasets/cadene/droid_1.0.1).

Convert a raw DROID folder directly into the format used by WEAVER:

```bash
python datasets/preprocess_droid.py --data_root /path/to/raw_droid
```

The script reads DROID metadata, Parquet trajectories, and camera videos. It
creates resized videos, SD3 latents, CLIP instruction features, and trajectory
annotations under `/path/to/raw_droid/weaver_preprocessed`:

```text
weaver_droid/
├── annotations/
│   └── <split>/<trajectory_id>.json
├── videos/
│   └── <split>/<trajectory_id>.mp4
├── latents/
│   └── <split>/<trajectory_id>.npz
└── done/
    └── <split>/<trajectory_id>
```

Test the pipeline on a few trajectories:

```bash
python datasets/preprocess_droid.py \
  --data_root /path/to/raw_droid \
  --max_trajectories 4
```

Set `--output_root /path/to/weaver_droid` to write somewhere else.

Shard preprocessing across jobs:

```bash
DATA_ROOT=/path/to/raw_droid \
OUTPUT_ROOT=/path/to/weaver_droid \
sbatch datasets/preprocess_droid.sh
```

After preprocessing, compute state/action normalization statistics:

```bash
python datasets/compute_norm_stats.py \
  --data_root /path/to/weaver_droid
```

## Preprocessing Customized OOD Data

To preprocess your own collected task data (multiple source roots, each with
task sub-directories containing `annotations/` and `videos/`) into the WEAVER
format with SD3 latents and CLIP text features:

```bash
python -m datasets.preprocess_droid_ood \
  --input_roots /path/to/ood_data /path/to/ood_data_add \
  --output_root /path/to/ood_data_weaver \
  --data_type train
```

The script produces the same layout as `preprocess_droid.py`:

```text
world_model_data_weaver/
├── annotations/
│   └── <data_type>/<id>.json    (includes text_features + latent_path)
├── videos/
│   └── <data_type>/<id>.mp4    (192x960, 3 cameras concatenated)
├── latents/
│   └── <data_type>/<id>_sd3.npz    (shape: 3 x T x 60 x 256, float16)
└── done/
    └── <data_type>/<id>    (marker files for resumability)
```

Run for both splits and then compute normalization statistics:

```bash
python -m datasets.preprocess_droid_ood \
  --input_roots /path/to/ood_data /path/to/ood_data_add \
  --output_root /path/to/ood_data_weaver \
  --data_type train

python -m datasets.preprocess_droid_ood \
  --input_roots /path/to/ood_data /path/to/ood_data_add \
  --output_root /path/to/ood_data_weaver \
  --data_type val

python datasets/compute_norm_stats.py \
  --data_root /path/to/ood_data_weaver
```

To process only specific tasks, pass `--tasks`:

```bash
python -m datasets.preprocess_droid_ood \
  --input_roots /path/to/ood_data \
  --output_root /path/to/ood_data_weaver \
  --data_type train \
  --tasks pour_task stack_task
```

---

## Reward Labeling with RoboMeter

WEAVER uses [RoboMeter-4B](https://huggingface.co/robometer/Robometer-4B)
to label task progress and success offline. RoboMeter is included as the
[`third_party/robometer`](../third_party/robometer) submodule.

Set up RoboMeter in its own environment:

```bash
cd third_party/robometer
uv sync
```

The batch client can label any dataset organized in the WEAVER format:

```text
<dataset_root>/
├── annotations/<split>/<trajectory_id>.json
└── videos/<split>/<trajectory_id>.mp4
```

Each annotation must provide the task instruction in `texts[0]`. Alternatively,
pass `--task` to use one instruction for all selected videos.

For each trajectory, the labeling pipeline:
1. Loads the trajectory video. For a video containing multiple concatenated
   camera views, pass `--view` to select the view used for reward prediction.
2. Reads the task instruction from `annotation["texts"][0]`, unless overridden
   with `--task`.
3. Sends the selected frames and instruction to the RoboMeter evaluation
   server. Frame-step expansion is disabled, so each trajectory uses one
   full-trajectory inference.
4. Receives two per-frame signals:
   - `reward_progress`: predicted task progress.
   - `reward_success`: predicted success probability.
5. Linearly interpolates both signals back to the original video length.
6. Computes the per-frame binary success label:

   ```python
   reward_binary = int(reward_success > 0.5)
   ```

7. Writes all three reward signals to the enriched annotations while preserving
   the original annotation fields:

   ```text
   annotation_rewards/
   ├── train/<trajectory_id>.json
   └── val/<trajectory_id>.json
   ```

For videos concatenated as `right | left | wrist`, use `--view right`.
Otherwise, omit `--view` or select the appropriate supported camera view.

The WEAVER-format batch client is implemented in
[`third_party/robometer/scripts/example_inference_droid_batch.py`](../third_party/robometer/scripts/example_inference_droid_batch.py).

Start the RoboMeter server:

```bash
cd third_party/robometer
uv run python robometer/evals/eval_server.py \
  server_url=0.0.0.0 \
  server_port=8000
```

Then label a dataset split from another terminal:

```bash
cd third_party/robometer
uv run python scripts/example_inference_droid_batch.py \
  --eval-server-url http://localhost:8000 \
  --data-root /path/to/dataset \
  --split train
```

For concatenated videos, add `--view right`. Repeat the client command
with `--split val` to label the validation split.

During training, the dataloader reads `reward_progress` from
`annotation_rewards/<split>/` by default.

With `dataset.negative_reward=True`, the dataloader applies:

```python
reward = reward_progress - 1.0
```

RoboMeter progress in `[0, 1]` therefore becomes a WEAVER reward in `[-1, 0]`.
The current loading behavior is defined in
[`weaver/datasets/droid.py`](../weaver/datasets/droid.py).
