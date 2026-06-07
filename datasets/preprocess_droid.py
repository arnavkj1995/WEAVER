#!/usr/bin/env python3
"""Convert a raw DROID release into the dataset format consumed by WEAVER."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from einops import rearrange
from tqdm import tqdm


DROID_CAMERAS = ["exterior_1_left", "exterior_2_left", "wrist_left"]


def read_jsonl(path: Path) -> List[Dict]:
    with path.open("r") as f:
        return [json.loads(line) for line in f if line.strip()]


def to_list(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value) if isinstance(value, tuple) else value


def parquet_column_to_list(frame, column: str) -> List:
    if column not in frame:
        raise KeyError(f"Required DROID parquet column not found: {column}")
    return [to_list(value) for value in frame[column]]


def read_resized_video(
    path: Path,
    image_size: Tuple[int, int],
    rgb_skip: int,
) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open DROID video: {path}")

    frames = []
    frame_idx = 0
    height, width = image_size
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % rgb_skip == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(
                cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
            )
        frame_idx += 1
    cap.release()

    if not frames:
        raise ValueError(f"No frames decoded from DROID video: {path}")
    return np.stack(frames, axis=0)


def encode_camera_frames(
    encoder: torch.nn.Module,
    frames: np.ndarray,
    batch_size: int,
    device: str,
) -> np.ndarray:
    frame_tensor = torch.from_numpy(
        rearrange(frames.astype(np.float32) / 255.0, "t h w c -> t c h w")
    )
    encoded = []
    with torch.no_grad():
        for start in range(0, len(frame_tensor), batch_size):
            batch = frame_tensor[start : start + batch_size].to(device)
            encoded.append(encoder(batch).cpu().numpy())
    return np.concatenate(encoded, axis=0)


def preprocess_droid(
    data_root: str,
    output_root: str,
    batch_size: int = 4,
    image_size: Tuple[int, int] = (192, 320),
    rgb_skip: int = 3,
    source_fps: float = 15.0,
    device: str = "cuda",
    chunks: int = 1,
    chunk_id: int = 0,
    max_trajectories: Optional[int] = None,
    use_fp16: bool = True,
    val_modulo: int = 100,
    val_remainder: int = 99,
) -> None:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("DROID preprocessing requires pandas and pyarrow.") from exc

    from weaver.wm.encoders import SD3Encoder, get_task_encoder

    if rgb_skip < 1:
        raise ValueError("rgb_skip must be at least 1")
    if chunks < 1 or not 0 <= chunk_id < chunks:
        raise ValueError(f"chunk_id must be in [0, {chunks}), got {chunk_id}")

    source_root = Path(data_root)
    output_root = Path(output_root)
    episodes_path = source_root / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        raise FileNotFoundError(f"DROID metadata not found: {episodes_path}")

    episodes = read_jsonl(episodes_path)
    if max_trajectories is not None:
        episodes = episodes[:max_trajectories]

    chunk_size = (len(episodes) + chunks - 1) // chunks
    start = chunk_id * chunk_size
    end = min(start + chunk_size, len(episodes))
    episodes = episodes[start:end]

    print("Loading SD3 and CLIP encoders...")
    image_encoder = SD3Encoder(
        model_name="stabilityai/stable-diffusion-3-medium-diffusers",
        image_size=image_size,
        spatial_size=4,
        device=device,
    ).eval()
    text_encoder = get_task_encoder(config={}, device=device).to(device).eval()

    print(
        f"Processing chunk {chunk_id + 1}/{chunks}: "
        f"episodes {start}-{end - 1} ({len(episodes)} trajectories)"
    )
    processed, skipped, errors = 0, 0, 0
    for episode in tqdm(episodes):
        trajectory_id = int(episode["episode_index"])
        split = "val" if trajectory_id % val_modulo == val_remainder else "train"
        done_path = output_root / "done" / split / str(trajectory_id)
        if done_path.exists():
            skipped += 1
            continue

        try:
            droid_chunk = trajectory_id // 1000
            parquet_path = (
                source_root
                / "data"
                / f"chunk-{droid_chunk:03d}"
                / f"episode_{trajectory_id:06d}.parquet"
            )
            frame = pd.read_parquet(parquet_path)
            instruction = episode["tasks"][0]

            camera_videos = []
            for camera in DROID_CAMERAS:
                camera_path = (
                    source_root
                    / "videos"
                    / f"chunk-{droid_chunk:03d}"
                    / f"observation.images.{camera}"
                    / f"episode_{trajectory_id:06d}.mp4"
                )
                camera_videos.append(
                    read_resized_video(camera_path, image_size, rgb_skip)
                )

            num_frames = min(len(video) for video in camera_videos)
            camera_videos = [video[:num_frames] for video in camera_videos]

            video_relpath = Path("videos") / split / f"{trajectory_id}.mp4"
            video_path = output_root / video_relpath
            video_path.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimwrite(
                video_path,
                np.concatenate(camera_videos, axis=2),
                fps=source_fps / rgb_skip,
                codec="libx264",
            )

            camera_latents = [
                encode_camera_frames(
                    image_encoder,
                    video,
                    batch_size=batch_size,
                    device=device,
                )
                for video in camera_videos
            ]
            latents = np.stack(camera_latents, axis=0)
            if use_fp16:
                latents = latents.astype(np.float16)

            latent_relpath = (
                Path("latents_sd3") / split / f"{trajectory_id}.npz"
            )
            latent_path = output_root / latent_relpath
            latent_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(latent_path, latents=latents)

            with torch.no_grad():
                text_features = (
                    text_encoder([instruction]).cpu().numpy()[0].tolist()
                )

            cartesian_position = parquet_column_to_list(
                frame, "observation.state.cartesian_position"
            )
            gripper_position = parquet_column_to_list(
                frame, "observation.state.gripper_position"
            )
            cartesian_states = np.concatenate(
                [
                    np.asarray(cartesian_position),
                    np.asarray(gripper_position).reshape(
                        len(gripper_position), -1
                    ),
                ],
                axis=-1,
            )[::rgb_skip][:num_frames]

            annotation = {
                "texts": [instruction],
                "text_features": text_features,
                "episode_id": trajectory_id,
                "success": int(frame["is_episode_successful"].iloc[0]),
                "video_length": num_frames,
                "state_length": len(cartesian_states),
                "raw_length": len(frame),
                "video_path": video_relpath.as_posix(),
                "sd3_latent_path": latent_relpath.as_posix(),
                "num_cameras": len(DROID_CAMERAS),
                "states": cartesian_states.tolist(),
                "observation.state.cartesian_position": cartesian_position,
                "observation.state.joint_position": parquet_column_to_list(
                    frame, "observation.state.joint_position"
                ),
                "observation.state.gripper_position": gripper_position,
                "action.cartesian_position": parquet_column_to_list(
                    frame, "action.cartesian_position"
                ),
                "action.joint_position": parquet_column_to_list(
                    frame, "action.joint_position"
                ),
                "action.gripper_position": parquet_column_to_list(
                    frame, "action.gripper_position"
                ),
                "action.joint_velocity": parquet_column_to_list(
                    frame, "action.joint_velocity"
                ),
            }
            annotation_path = (
                output_root
                / "annotation_rewards"
                / split
                / f"{trajectory_id}.json"
            )
            annotation_path.parent.mkdir(parents=True, exist_ok=True)
            with annotation_path.open("w") as f:
                json.dump(annotation, f, indent=2)

            done_path.parent.mkdir(parents=True, exist_ok=True)
            done_path.touch()
            processed += 1
        except Exception as exc:
            print(f"Error processing DROID trajectory {trajectory_id}: {exc}")
            errors += 1

    print(f"Processed {processed}, skipped {skipped}, errors {errors}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a raw DROID folder into WEAVER videos, annotations, "
            "SD3 latents, and CLIP text features."
        )
    )
    parser.add_argument("--data_root", required=True, help="Raw DROID root")
    parser.add_argument(
        "--output_root",
        default=None,
        help="WEAVER dataset root (default: <data_root>/weaver_preprocessed)",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--image_height", type=int, default=192)
    parser.add_argument("--image_width", type=int, default=320)
    parser.add_argument("--rgb_skip", type=int, default=3)
    parser.add_argument("--source_fps", type=float, default=15.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunks", type=int, default=1)
    parser.add_argument("--chunk_id", type=int, default=0)
    parser.add_argument("--max_trajectories", type=int, default=None)
    parser.add_argument("--use_fp16", action="store_true", default=True)
    parser.add_argument(
        "--no_fp16", dest="use_fp16", action="store_false"
    )
    args = parser.parse_args()
    output_root = args.output_root or str(
        Path(args.data_root) / "weaver_preprocessed"
    )

    preprocess_droid(
        data_root=args.data_root,
        output_root=output_root,
        batch_size=args.batch_size,
        image_size=(args.image_height, args.image_width),
        rgb_skip=args.rgb_skip,
        source_fps=args.source_fps,
        device=args.device,
        chunks=args.chunks,
        chunk_id=args.chunk_id,
        max_trajectories=args.max_trajectories,
        use_fp16=args.use_fp16,
    )


if __name__ == "__main__":
    main()
