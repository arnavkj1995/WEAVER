"""Preprocessing utilities for DROID-style WEAVER datasets.

This file contains offline utilities for normalization statistics, language
feature encoding, latent conversion, video frame caching, and SD3 latent
encoding. Runtime dataset loading lives in ``weaver.datasets.droid``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from einops import rearrange
from torch.utils.data import DataLoader
from tqdm import tqdm

from weaver.datasets.droid import PrecomputedDroid


RGB_SKIP = 3


def _annotation_dir(data_root: str, split: str) -> Path:
    anno_dir = Path(data_root) / f"annotations/{split}"
    if not anno_dir.exists():
        anno_dir = Path(data_root) / f"annotation_rewards/{split}"
    return anno_dir


def preprocess_states_actions(annotation: Dict, relabel_actions: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    joint_position = np.array(annotation["observation.state.joint_position"])
    gripper_position = np.array(annotation["observation.state.gripper_position"])[:, None]
    if len(gripper_position.shape) == 3:
        gripper_position = gripper_position[:, 0, :]
    full_states = np.concatenate([joint_position, gripper_position], axis=-1)

    state_indices = np.arange(0, len(joint_position), RGB_SKIP)
    states = full_states[state_indices]

    if relabel_actions:
        next_state_indices = np.clip(state_indices + RGB_SKIP, 0, len(full_states) - 1)
        actions = full_states[next_state_indices] - full_states[state_indices]
    else:
        action_joint = np.array(annotation["action.joint_velocity"])
        action_gripper = np.array(annotation["action.gripper_position"])
        if action_gripper.ndim == 1:
            action_gripper = action_gripper[:, None]

        actions = []
        for i in state_indices:
            end_action_idx = min(i + RGB_SKIP, len(action_joint))
            action_sum = action_joint[i:end_action_idx].sum(axis=0)
            gripper_last = action_gripper[end_action_idx - 1]
            actions.append(np.concatenate([action_sum, gripper_last], axis=-1))
        actions = np.stack(actions, axis=0)

    return states, actions


def compute_norm_stats(
    data_root: str,
    output_path: Optional[str] = None,
    max_trajectories: Optional[int] = None,
    relabel_actions: bool = True,
) -> Dict:
    if output_path is None:
        suffix = "relabel" if relabel_actions else "recorded"
        output_path = os.path.join(data_root, f"norm_stats_{suffix}.json")

    anno_dir = _annotation_dir(data_root, "train")
    if not anno_dir.exists():
        raise ValueError(f"Annotation directory not found: {anno_dir}")

    anno_files = sorted(anno_dir.glob("*.json"))
    if max_trajectories:
        anno_files = anno_files[:max_trajectories]

    all_states, all_actions = [], []
    print(f"Computing normalization stats from {len(anno_files)} trajectories...")
    for anno_file in tqdm(anno_files):
        with anno_file.open("r") as f:
            annotation = json.load(f)
        states, actions = preprocess_states_actions(annotation, relabel_actions=relabel_actions)
        all_states.append(states)
        all_actions.append(actions)

    all_states = np.concatenate(all_states, axis=0)
    all_actions = np.concatenate(all_actions, axis=0)
    norm_stats = {
        "norm_stats": {
            "state": {
                "mean": all_states.mean(axis=0).tolist(),
                "std": all_states.std(axis=0).tolist(),
            },
            "actions": {
                "mean": all_actions.mean(axis=0).tolist(),
                "std": all_actions.std(axis=0).tolist(),
            },
        }
    }

    with open(output_path, "w") as f:
        json.dump(norm_stats, f, indent=2)
    print(f"Saved normalization stats to {output_path}")
    return norm_stats


def encode_text_features(
    data_root: str,
    text_encoder: torch.nn.Module,
    splits: List[str] = ["train", "val"],
    batch_size: int = 64,
) -> None:
    for split in splits:
        anno_dir = _annotation_dir(data_root, split)
        if not anno_dir.exists():
            print(f"Skipping {split} split - directory not found: {anno_dir}")
            continue

        anno_files = sorted(anno_dir.glob("*.json"))
        files_to_process = []
        for anno_file in anno_files:
            with anno_file.open("r") as f:
                annotation = json.load(f)
            if "text_features" not in annotation:
                files_to_process.append(anno_file)

        if not files_to_process:
            print(f"All {split} trajectories already have text features, skipping.")
            continue

        print(f"Encoding text features for {len(files_to_process)} {split} trajectories...")
        with torch.no_grad():
            for i in tqdm(range(0, len(files_to_process), batch_size)):
                batch_files = files_to_process[i : i + batch_size]
                batch_texts, batch_annotations = [], []
                for anno_file in batch_files:
                    with anno_file.open("r") as f:
                        annotation = json.load(f)
                    batch_texts.append(annotation["texts"][0])
                    batch_annotations.append(annotation)

                features = text_encoder(batch_texts).cpu().numpy()
                for anno_file, annotation, feat in zip(batch_files, batch_annotations, features):
                    annotation["text_features"] = feat.tolist()
                    with anno_file.open("w") as f:
                        json.dump(annotation, f, indent=2)


def convert_pt_to_npy(
    data_root: str,
    splits: List[str] = ["train", "val"],
    delete_pt: bool = False,
) -> None:
    for split in splits:
        anno_dir = _annotation_dir(data_root, split)
        if not anno_dir.exists():
            print(f"Skipping {split} split - directory not found: {anno_dir}")
            continue

        converted, skipped = 0, 0
        for anno_file in tqdm(sorted(anno_dir.glob("*.json"))):
            with anno_file.open("r") as f:
                annotation = json.load(f)

            pt_path = os.path.join(data_root, annotation["latent_path"])
            npy_path = pt_path.replace(".pt", ".npy")
            if os.path.exists(npy_path):
                skipped += 1
                continue
            if not os.path.exists(pt_path):
                print(f"Warning: {pt_path} not found, skipping")
                continue

            latents = torch.load(pt_path, weights_only=True).numpy()
            np.save(npy_path, latents)
            annotation["latent_path"] = annotation["latent_path"].replace(".pt", ".npy")
            with anno_file.open("w") as f:
                json.dump(annotation, f, indent=2)
            if delete_pt:
                os.remove(pt_path)
            converted += 1

        print(f"Converted {converted} files, skipped {skipped} existing files")


def _load_video_as_array(video_path: str, num_cameras: int) -> Optional[np.ndarray]:
    if not os.path.exists(video_path):
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames:
        return None

    stacked_video = np.stack(frames, axis=0)
    camera_width = stacked_video.shape[2] // num_cameras
    video_by_camera = []
    for i in range(num_cameras):
        start_w = i * camera_width
        end_w = (i + 1) * camera_width
        video_by_camera.append(stacked_video[:, :, start_w:end_w, :])
    return np.stack(video_by_camera, axis=0)


def convert_videos_to_npy(
    data_root: str,
    splits: List[str] = ["train", "val"],
    delete_video: bool = False,
) -> None:
    for split in splits:
        anno_dir = _annotation_dir(data_root, split)
        if not anno_dir.exists():
            print(f"Skipping {split} split - directory not found: {anno_dir}")
            continue

        converted, skipped, errors = 0, 0, 0
        for anno_file in tqdm(sorted(anno_dir.glob("*.json"))):
            try:
                with anno_file.open("r") as f:
                    annotation = json.load(f)

                video_path = os.path.join(data_root, annotation["video_path"])
                npy_path = video_path.replace(".mp4", "_frames.npy")
                if os.path.exists(npy_path):
                    skipped += 1
                    continue

                video_array = _load_video_as_array(video_path, annotation["num_cameras"])
                if video_array is None:
                    print(f"Warning: could not load {video_path}")
                    errors += 1
                    continue

                np.save(npy_path, video_array)
                annotation["video_frames_path"] = annotation["video_path"].replace(".mp4", "_frames.npy")
                with anno_file.open("w") as f:
                    json.dump(annotation, f, indent=2)
                if delete_video:
                    os.remove(video_path)
                converted += 1
            except Exception as exc:
                print(f"Error processing {anno_file}: {exc}")
                errors += 1

        print(f"Converted {converted} videos, skipped {skipped}, errors {errors}")


def encode_sd3_features(
    data_root: str,
    splits: List[str] = ["train", "val"],
    batch_size: int = 8,
    image_size: Tuple[int, int] = (192, 320),
    device: str = "cuda",
    chunks: int = 1,
    chunk_id: int = 0,
    use_fp16: bool = True,
) -> None:
    from weaver.wm.encoders import SD3Encoder

    print("Loading SD3 encoder...")
    encoder = SD3Encoder(
        model_name="stabilityai/stable-diffusion-3-medium-diffusers",
        image_size=image_size,
        spatial_size=4,
        device=device,
    )
    encoder.eval()

    for split in splits:
        anno_dir = _annotation_dir(data_root, split)
        if not anno_dir.exists():
            print(f"Skipping {split} split - directory not found: {anno_dir}")
            continue

        anno_files = sorted(anno_dir.glob("*.json"))
        total_files = len(anno_files)
        if chunks > 1:
            chunk_size = (total_files + chunks - 1) // chunks
            start_idx = chunk_id * chunk_size
            end_idx = min(start_idx + chunk_size, total_files)
            anno_files = anno_files[start_idx:end_idx]
            print(f"Processing chunk {chunk_id}/{chunks}: {start_idx}-{end_idx} of {total_files}")

        encoded, skipped, errors = 0, 0, 0
        for anno_file in tqdm(anno_files):
            try:
                with anno_file.open("r") as f:
                    annotation = json.load(f)

                if "latent_path" not in annotation:
                    annotation["latent_path"] = f"latents/{split}/{annotation['episode_id']}.npy"
                    with anno_file.open("w") as f:
                        json.dump(annotation, f, indent=2)

                latent_path = os.path.join(data_root, annotation["latent_path"])
                sd3_latent_path = (
                    latent_path.replace(".npy", "_sd3.npz")
                    if latent_path.endswith(".npy")
                    else latent_path.replace(".pt", "_sd3.npz")
                )
                if os.path.exists(sd3_latent_path):
                    skipped += 1
                    continue

                frames = _load_video_as_array(
                    os.path.join(data_root, annotation["video_path"]),
                    annotation["num_cameras"],
                )
                if frames is None:
                    print(f"Warning: could not load frames for {anno_file}")
                    errors += 1
                    continue

                all_camera_latents = []
                for cam_idx in range(annotation["num_cameras"]):
                    cam_frames = frames[cam_idx].astype(np.float32) / 255.0
                    cam_frames = torch.from_numpy(rearrange(cam_frames, "t h w c -> t c h w")).to(device)
                    latents_list = []
                    for i in range(0, cam_frames.shape[0], batch_size):
                        with torch.no_grad():
                            latents = encoder(cam_frames[i : i + batch_size])
                        latents_list.append(latents.cpu().numpy())
                    all_camera_latents.append(np.concatenate(latents_list, axis=0))

                stacked_latents = np.stack(all_camera_latents, axis=0)
                if use_fp16:
                    stacked_latents = stacked_latents.astype(np.float16)
                np.savez_compressed(sd3_latent_path, latents=stacked_latents)
                encoded += 1
            except Exception as exc:
                print(f"Error processing {anno_file}: {exc}")
                errors += 1

        print(f"Encoded {encoded} trajectories, skipped {skipped}, errors {errors}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        default="test",
        choices=["test", "compute_norm_stats", "encode_text", "convert_to_npy", "convert_videos", "encode_sd3"],
    )
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "valid"])
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_trajectories", type=int, default=None)
    parser.add_argument("--relabel_actions", action="store_true", default=True)
    parser.add_argument("--no_relabel_actions", dest="relabel_actions", action="store_false")
    parser.add_argument("--chunks", type=int, default=1)
    parser.add_argument("--chunk_id", type=int, default=0)
    parser.add_argument("--use_fp16", action="store_true", default=True)
    parser.add_argument("--no_fp16", dest="use_fp16", action="store_false")
    args = parser.parse_args()

    if args.mode == "compute_norm_stats":
        compute_norm_stats(args.data_root, max_trajectories=args.max_trajectories, relabel_actions=args.relabel_actions)
        return
    if args.mode == "encode_text":
        from weaver.wm.encoders import get_task_encoder

        text_encoder = get_task_encoder(config={}, device="cuda").to("cuda")
        encode_text_features(args.data_root, text_encoder, splits=["train", "val"], batch_size=args.batch_size)
        return
    if args.mode == "convert_to_npy":
        convert_pt_to_npy(args.data_root, splits=["train", "val"])
        return
    if args.mode == "convert_videos":
        convert_videos_to_npy(args.data_root, splits=["train", "val"])
        return
    if args.mode == "encode_sd3":
        encode_sd3_features(
            args.data_root,
            splits=[args.split],
            batch_size=args.batch_size,
            chunks=args.chunks,
            chunk_id=args.chunk_id,
            use_fp16=args.use_fp16,
        )
        return

    dataset = PrecomputedDroid(
        root=args.data_root,
        split=args.split,
        horizon=args.horizon,
        img_keys=["exterior_1_left", "wrist_left", "exterior_2_left"],
        relabel_actions=args.relabel_actions,
        normalize=True,
        cache_trajectories=False,
        return_language=True,
        return_video_frames=True,
        max_trajectories=args.max_trajectories,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    print(f"Dataset size: {len(dataset)} trajectories")
    print(f"Dataloader batches: {len(dataloader)}")

    for batch_idx, batch in enumerate(dataloader):
        print(f"Batch {batch_idx}: obs={list(batch['obs'].keys())}, actions={batch['actions'].shape}")
        if batch_idx >= 2:
            break


if __name__ == "__main__":
    main()
