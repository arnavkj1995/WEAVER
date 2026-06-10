#!/usr/bin/env python3
"""Preprocess our collected OOD task data into the WEAVER dataset format.

Extends SAILOR-FM/datasets/preprocess_ours_all.py to also produce SD3 visual
latents and CLIP text features required by WEAVER (as in preprocess_droid.py).

Input layout (one or more input roots, each with task sub-directories):
    <input_root>/
        <task_name>/
            annotations/<id>.json
            videos/<id>.mp4

Output layout:
    <output_root>/
        videos/<data_type>/<id>.mp4          (192x960, left|right|wrist)
        annotations/<data_type>/<id>.json    (includes text_features + latent_path)
        latents/<data_type>/<id>_sd3.npz     (shape: 3 x T x 60 x 256, float16)

Pipeline per episode:
  1. Read source video (3 cams side-by-side: right|left|wrist)
  2. Apply frameskip (default: every 3rd frame)
  3. Resize full frame to 192x960 so each camera becomes 192x320
  4. Write output video @ 5 fps
  5. Encode each camera view through SD3 → save latents
  6. Encode instruction text through CLIP → save in annotation
  7. Write DROID-style annotation JSON

Example:
    python datasets/preprocess_droid_ood.py \\
        --input_roots /data/yilin/world_model_data /data/yilin/world_model_data_add \\
        --output_root /data/yilin/world_model_data_weaver \\
        --data_type val
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Optional

import mediapy
import numpy as np
import torch
from einops import rearrange
from tqdm import tqdm


TASKS = [
    "pour_task",
    "towel_task",
    "bag_task",
    "cup_task",
    "stack_task",
    "marker_task",
]

TASK_PREFIX = {
    "pour_task":   "pour_our",
    "towel_task":  "towel_our",
    "bag_task":    "bag_our",
    "cup_task":    "cup_our",
    "stack_task":  "stack_our",
    "marker_task": "marker_our",
}

CAM_H, CAM_W = 192, 320

RAW_KEYS = [
    "observation.state.cartesian_position",
    "observation.state.joint_position",
    "observation.state.gripper_position",
    "action.cartesian_position",
    "action.joint_position",
    "action.gripper_position",
    "action.joint_velocity",
]


def process_video_frames(
    src_path: Path,
    rgb_skip: int = 3,
    cam_order: list[int] = [0, 1, 2],
    max_frames: Optional[int] = None,
) -> np.ndarray:
    """Read source video, apply frameskip, resize, reorder cameras.

    Returns (T, H, 3*W, 3) uint8 RGB array.
    """
    video = mediapy.read_video(str(src_path))
    frames = torch.tensor(video).permute(0, 3, 1, 2).float() / 255.0 * 2 - 1
    frames = frames[::rgb_skip]

    x = torch.nn.functional.interpolate(
        frames, size=(CAM_H, CAM_W * 3), mode="bilinear", align_corners=False
    )
    resized = ((x / 2.0 + 0.5).clamp(0, 1) * 255)
    resized = resized.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

    panels = [resized[:, :, i * CAM_W:(i + 1) * CAM_W, :] for i in range(3)]
    reordered = np.concatenate([panels[i] for i in cam_order], axis=2)

    if max_frames is not None:
        reordered = reordered[:max_frames]

    return reordered


def write_video(frames: np.ndarray, dst_path: Path, fps: float = 5.0) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4", dir=dst_path.parent)
    os.close(tmp_fd)
    try:
        mediapy.write_video(tmp_path, frames, fps=fps)
        os.replace(tmp_path, str(dst_path))
    except Exception:
        os.unlink(tmp_path)
        raise


def encode_camera_frames(
    encoder: torch.nn.Module,
    frames: np.ndarray,
    batch_size: int,
    device: str,
) -> np.ndarray:
    """Encode (T, H, W, 3) uint8 frames through SD3. Returns (T, tokens, channels)."""
    frame_tensor = torch.from_numpy(
        rearrange(frames.astype(np.float32) / 255.0, "t h w c -> t c h w")
    )
    encoded = []
    with torch.no_grad():
        for start in range(0, len(frame_tensor), batch_size):
            batch = frame_tensor[start:start + batch_size].to(device)
            encoded.append(encoder(batch).cpu().numpy())
    return np.concatenate(encoded, axis=0)


def build_annotation(
    anno: dict,
    output_idx: int,
    episode_id_orig: str,
    video_length: int,
    rgb_skip: int,
    data_type: str,
    text_features: list,
) -> dict:
    texts = anno["texts"]
    if isinstance(texts, str):
        texts = [texts]

    raw_length = min(len(anno[k]) for k in RAW_KEYS)
    truncated = {k: anno[k][:raw_length] for k in RAW_KEYS}

    cartesian = np.array(truncated["observation.state.cartesian_position"])
    gripper = np.array(truncated["observation.state.gripper_position"])
    gripper = 1 - gripper / 0.085
    truncated["observation.state.gripper_position"] = gripper.tolist()
    full_states = np.concatenate([cartesian, gripper], axis=-1)
    states_sampled = full_states[::rgb_skip].tolist()[:video_length]

    return {
        "texts":           texts,
        "text_features":   text_features,
        "episode_id":      output_idx,
        "episode_id_orig": episode_id_orig,
        "success":         int(anno.get("success", 0)),
        "video_length":    video_length,
        "state_length":    len(states_sampled),
        "raw_length":      raw_length,
        "video_path":      f"videos/{data_type}/{output_idx}.mp4",
        "latent_path":     f"latents/{data_type}/{output_idx}_sd3.npz",
        "num_cameras":     int(anno.get("num_cameras", 3)),
        "states":          states_sampled,
        **truncated,
    }


def collect_episodes(
    input_roots: list[Path], tasks: list[str]
) -> dict[str, list[tuple[Path, int]]]:
    """Discover all (task_dir, ep_id) pairs across all input roots."""
    task_episodes: dict[str, list[tuple[Path, int]]] = defaultdict(list)
    for root in input_roots:
        for task in tasks:
            anno_dir = root / task / "annotations"
            if not anno_dir.exists():
                continue
            ep_ids = sorted(int(p.stem) for p in anno_dir.glob("*.json"))
            for ep_id in ep_ids:
                task_episodes[task].append((root / task, ep_id))
    return dict(task_episodes)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess OOD task data into WEAVER format with SD3 latents and CLIP text features."
    )
    parser.add_argument(
        "--input_roots", type=str, nargs="+", required=True,
        help="One or more source root directories, each containing task sub-directories.",
    )
    parser.add_argument(
        "--output_root", type=str, required=True,
        help="Root directory for output videos, annotations, and latents.",
    )
    parser.add_argument("--rgb_skip",   type=int,  default=3,
                        help="Keep every Nth frame (default: 3).")
    parser.add_argument("--data_type",  type=str,  default="val",
                        help="Output split name, e.g. 'train' or 'val'.")
    parser.add_argument("--batch_size", type=int,  default=4,
                        help="Batch size for SD3 encoder.")
    parser.add_argument("--device",     type=str,  default="cuda")
    parser.add_argument("--use_fp16",   action="store_true", default=True,
                        help="Save latents as float16 (default: True).")
    parser.add_argument("--no_fp16",    dest="use_fp16", action="store_false")
    parser.add_argument(
        "--tasks", type=str, nargs="*", default=None,
        help="Task sub-directory names to process. Default: auto-discover from input roots.",
    )
    args = parser.parse_args()

    input_roots = [Path(r) for r in args.input_roots]
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.tasks:
        tasks = args.tasks
    else:
        discovered: set[str] = set()
        for root in input_roots:
            if root.exists():
                discovered.update(p.name for p in root.iterdir() if p.is_dir())
        known = [t for t in TASKS if t in discovered]
        extra = sorted(discovered - set(TASKS))
        tasks = known + extra

    print(f"Input roots: {[str(r) for r in input_roots]}")
    print(f"Output:      {output_root}")
    print(f"Tasks:       {tasks}")
    print(f"Frame skip:  {args.rgb_skip}")
    print(f"Cam size:    {CAM_H} x {CAM_W}  →  full frame {CAM_H} x {CAM_W * 3}")
    print(f"FP16:        {args.use_fp16}\n")

    print("Loading SD3 and CLIP encoders...")
    from weaver.wm.encoders import SD3Encoder, get_task_encoder
    image_encoder = SD3Encoder(
        model_name="stabilityai/stable-diffusion-3-medium-diffusers",
        image_size=(CAM_H, CAM_W),
        spatial_size=4,
        device=args.device,
    ).eval()
    text_encoder = get_task_encoder(config={}, device=args.device).to(args.device).eval()
    print("Encoders ready.\n")

    task_episodes = collect_episodes(input_roots, tasks)

    total_ok = total_skip = output_idx = 0

    for task in tasks:
        episodes = task_episodes.get(task, [])
        if not episodes:
            print(f"{task}: no episodes found across all roots — skipping")
            continue

        prefix = TASK_PREFIX.get(task, task)
        print(f"{task}: {len(episodes)} episodes")
        local_idx = 0

        for task_dir, ep_id in tqdm(episodes, desc=task):
            src_video = task_dir / "videos"      / f"{ep_id}.mp4"
            src_anno  = task_dir / "annotations" / f"{ep_id}.json"

            if not src_video.exists() or not src_anno.exists():
                print(f"  Skip ep {ep_id} in {task_dir}: missing file(s)")
                total_skip += 1
                local_idx  += 1
                continue

            # Skip already-processed episodes
            done_path = output_root / "done" / args.data_type / str(output_idx)
            if done_path.exists():
                output_idx += 1
                local_idx  += 1
                total_ok   += 1
                continue

            dst_video  = output_root / "videos"      / args.data_type / f"{output_idx}.mp4"
            dst_anno   = output_root / "annotations" / args.data_type / f"{output_idx}.json"
            dst_latent = output_root / "latents"     / args.data_type / f"{output_idx}_sd3.npz"

            with open(src_anno) as f:
                anno = json.load(f)

            instruction = anno["texts"]
            if isinstance(instruction, list):
                instruction = instruction[0]

            raw_min = min(len(anno[k]) for k in RAW_KEYS)
            state_frames = len(np.arange(raw_min)[::args.rgb_skip])

            try:
                frames = process_video_frames(
                    src_video, rgb_skip=args.rgb_skip, max_frames=state_frames
                )
                video_length = len(frames)
                write_video(frames, dst_video)

                # Encode SD3 latents — one per camera view
                camera_frames = [frames[:, :, i * CAM_W:(i + 1) * CAM_W, :] for i in range(3)]
                camera_latents = [
                    encode_camera_frames(image_encoder, cam_f, args.batch_size, args.device)
                    for cam_f in camera_frames
                ]
                latents = np.stack(camera_latents, axis=0)  # (3, T, tokens, channels)
                if args.use_fp16:
                    latents = latents.astype(np.float16)
                dst_latent.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(dst_latent, latents=latents)

                # Encode CLIP text features
                with torch.no_grad():
                    text_features = text_encoder([instruction]).cpu().numpy()[0].tolist()

            except Exception as e:
                print(f"  Skip ep {ep_id} in {task_dir}: {e}")
                total_skip += 1
                local_idx  += 1
                continue

            episode_id_orig = f"{prefix}_{local_idx:03d}"
            out_anno = build_annotation(
                anno, output_idx, episode_id_orig,
                video_length, args.rgb_skip, args.data_type, text_features,
            )
            dst_anno.parent.mkdir(parents=True, exist_ok=True)
            with open(dst_anno, "w") as f:
                json.dump(out_anno, f, indent=2)

            done_path.parent.mkdir(parents=True, exist_ok=True)
            done_path.touch()

            total_ok   += 1
            output_idx += 1
            local_idx  += 1

    print(f"\nDone. Processed: {total_ok}, skipped: {total_skip}")
    print(f"Output indices: 0 to {output_idx - 1}")
    print(f"  {output_root}/videos/{args.data_type}/")
    print(f"  {output_root}/annotations/{args.data_type}/")
    print(f"  {output_root}/latents/{args.data_type}/")

    # Spot-check first 3 outputs
    print("\nSpot-checking first 3 outputs:")
    for i in range(min(3, output_idx)):
        vpath = output_root / "videos"      / args.data_type / f"{i}.mp4"
        apath = output_root / "annotations" / args.data_type / f"{i}.json"
        lpath = output_root / "latents"     / args.data_type / f"{i}_sd3.npz"
        try:
            v = mediapy.read_video(str(vpath))
            with open(apath) as f:
                d = json.load(f)
            latents = np.load(str(lpath))["latents"]
            frame_match = (
                len(v) == d["video_length"] == d["state_length"] == len(d["states"])
            )
            latent_match = latents.shape == (3, len(v), latents.shape[2], latents.shape[3])
            status = "OK" if frame_match and latent_match else "MISMATCH"
            print(
                f"  {i}: frames={len(v)}  latents={latents.shape}"
                f"  ep_orig={d['episode_id_orig']}  [{status}]"
            )
        except Exception as e:
            print(f"  {i}: spot-check failed: {e}")


if __name__ == "__main__":
    main()
