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

TODO: Add reward annotations generated with RoboMeter.

---

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
