"""
Training script for the Franka dynamics model.

Usage:
    python -m weaver.dynamics.train
"""

import json
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from .model import Dynamics


class DynamicsDataset(Dataset):
    """Loads (joint, gripper) trajectories from DROID-format annotation files."""

    def __init__(self, data_root: str, num_frames: int, mode: str = "train"):
        self.data_root = data_root
        self.num_frames = num_frames

        ann_dir = os.path.join(data_root, "annotations", mode)
        ann_files = sorted(os.listdir(ann_dir), key=lambda x: int(x.split(".")[0]))

        self.labels: dict = {}
        self.samples: list = []

        for ann_file in ann_files:
            ep_idx = int(ann_file.split(".")[0])
            with open(os.path.join(ann_dir, ann_file)) as f:
                label = json.load(f)
            self.labels[ep_idx] = label
            for start_id in range(label["raw_length"] - num_frames):
                self.samples.append((ep_idx, start_id))

        print(f"DynamicsDataset ({mode}): {len(self.samples)} segments from {len(self.labels)} episodes")

    def __len__(self) -> int:
        return len(self.samples)

    def _fetch(self, index: int) -> dict:
        ep_idx, start_id = self.samples[index]
        label = self.labels[ep_idx]
        max_id = label["raw_length"] - 1
        ids = np.clip(np.arange(start_id, start_id + self.num_frames + 1), 0, max_id)

        joints       = np.array(label["observation.state.joint_position"],  dtype=np.float32)[ids]
        joint_vels   = np.array(label["action.joint_velocity"],              dtype=np.float32)[ids]
        gripper_pos  = np.array(label["observation.state.gripper_position"], dtype=np.float32)[ids].reshape(-1, 1)
        gripper_acts = np.array(label["action.gripper_position"],            dtype=np.float32)[ids].reshape(-1, 1)

        return {
            "joints":          joints[0:1],
            "joint_vels":      joint_vels[:-1],
            "joints_delta":    joints[1:] - joints[0:1],
            "gripper":         gripper_pos[0:1],
            "gripper_actions": gripper_acts[:-1],
            "gripper_delta":   gripper_pos[1:] - gripper_pos[0:1],
        }

    def __getitem__(self, index: int) -> dict:
        try:
            return self._fetch(index)
        except Exception:
            return self._fetch(random.randint(0, len(self.samples) - 1))


if __name__ == "__main__":
    DATA_ROOT  = "/data/yilin/world_model_data_our_50"
    ACTION_DIM = 7
    NUM_FRAMES = 15
    EPOCHS     = 15
    BATCH_SIZE = 128

    dataset    = DynamicsDataset(DATA_ROOT, num_frames=NUM_FRAMES, mode="train")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=16)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = Dynamics(action_dim=ACTION_DIM, action_num=NUM_FRAMES, hidden_size=512).to(device)
    model.device = device
    opt    = torch.optim.Adam(model.parameters(), lr=1e-4)

    out_dir = "output_dynamics_all_250_gripper_weight_5_binarized"
    os.makedirs(out_dir, exist_ok=True)

    step = 0
    running_loss = 0.0
    for epoch in range(EPOCHS):
        for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
            loss = model(
                batch["joints"], batch["joint_vels"], batch["joints_delta"],
                batch["gripper"], batch["gripper_actions"], batch["gripper_delta"],
                training=True,
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            running_loss += loss.item()
            if step % 100 == 0:
                print(f"step {step}  loss {running_loss / 100:.4f}")
                running_loss = 0.0

        torch.save(model.state_dict(), os.path.join(out_dir, f"model2_{NUM_FRAMES}_{epoch}.pth"))
        print(f"Epoch {epoch} done — checkpoint saved.")
