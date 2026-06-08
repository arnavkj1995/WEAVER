#!/usr/bin/env python3
"""Compute state/action normalization stats for a preprocessed dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from tqdm import tqdm


RGB_SKIP = 3


def annotation_dir(data_root: str, split: str) -> Path:
    return Path(data_root) / "annotations" / split


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
    suffix = "relabel" if relabel_actions else "recorded"
    output = Path(output_path) if output_path else Path(data_root) / f"norm_stats_{suffix}.json"

    anno_dir = annotation_dir(data_root, "train")
    if not anno_dir.exists():
        raise FileNotFoundError(f"Annotation directory not found: {anno_dir}")

    anno_files = sorted(anno_dir.glob("*.json"))
    if max_trajectories is not None:
        anno_files = anno_files[:max_trajectories]
    if not anno_files:
        raise ValueError(f"No annotation files found in {anno_dir}")

    all_states, all_actions = [], []
    print(f"Computing normalization stats from {len(anno_files)} train trajectories...")
    for anno_file in tqdm(anno_files):
        with anno_file.open("r") as f:
            annotation = json.load(f)
        states, actions = preprocess_states_actions(annotation, relabel_actions=relabel_actions)
        all_states.append(states)
        all_actions.append(actions)

    states = np.concatenate(all_states, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    norm_stats = {
        "norm_stats": {
            "state": {
                "mean": states.mean(axis=0).tolist(),
                "std": states.std(axis=0).tolist(),
            },
            "actions": {
                "mean": actions.mean(axis=0).tolist(),
                "std": actions.std(axis=0).tolist(),
            },
        }
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        json.dump(norm_stats, f, indent=2)
    print(f"Saved normalization stats to {output}")
    return norm_stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute WEAVER DROID normalization stats.")
    parser.add_argument("--data_root", type=str, required=True, help="Path to the preprocessed dataset root.")
    parser.add_argument("--output_path", type=str, default=None, help="Optional output JSON path.")
    parser.add_argument("--max_trajectories", type=int, default=None)
    parser.add_argument("--relabel_actions", action="store_true", default=True)
    parser.add_argument("--no_relabel_actions", dest="relabel_actions", action="store_false")
    args = parser.parse_args()

    compute_norm_stats(
        data_root=args.data_root,
        output_path=args.output_path,
        max_trajectories=args.max_trajectories,
        relabel_actions=args.relabel_actions,
    )


if __name__ == "__main__":
    main()
