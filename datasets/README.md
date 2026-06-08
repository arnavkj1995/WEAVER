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
python datasets/preprocess_droid.py \
  --data_root /path/to/raw_droid \
  --output_root /path/to/weaver_droid \
  --chunks 8 \
  --chunk_id 0
```

After preprocessing, compute state/action normalization statistics:

```bash
python datasets/compute_norm_stats.py \
  --data_root /path/to/weaver_droid
```

TODO: Add reward annotations generated with RoboMeter.
