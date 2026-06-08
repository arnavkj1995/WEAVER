"""DROID-style dataset loader for WEAVER world-model training and evaluation."""

import os
import json
import random
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from einops import rearrange


CAMERA_NAMES = ['exterior_1_left', 'exterior_2_left', 'wrist_left']
RGB_SKIP = 3


def read_video_frames_range(video_path: str, start_idx: int, end_idx: int, num_cameras: int) -> Dict[str, np.ndarray]:
    """Read a specific range of frames from a stacked video using random access.

    This is much faster than reading the entire video when only a small chunk is needed.
    Uses OpenCV's seek capability to jump directly to the start frame.

    Args:
        video_path: Path to the video file
        start_idx: Starting frame index (inclusive)
        end_idx: Ending frame index (exclusive)
        num_cameras: Number of cameras stacked horizontally in the video

    Returns:
        Dictionary mapping camera names to numpy arrays of shape (num_frames, H, W, C)
    """
    camera_names = CAMERA_NAMES[:num_cameras]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    camera_width = frame_width // num_cameras

    frames_by_camera = {cam: [] for cam in camera_names}

    for _ in range(end_idx - start_idx):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        for i, cam_name in enumerate(camera_names):
            start_w = i * camera_width
            end_w = (i + 1) * camera_width
            frames_by_camera[cam_name].append(frame[:, start_w:end_w, :])

    cap.release()

    result = {}
    for cam_name, frames in frames_by_camera.items():
        if frames:
            result[cam_name] = np.stack(frames, axis=0)

    return result


class DROIDTrajectory:
    """Container for a single DROID trajectory"""

    def __init__(self, data_root: str, traj_id: int, data_type: str, load_on_init: bool = False, encoder_type: str = "svd", annotation_dir: str = "annotations"):
        self.data_root = data_root
        self.traj_id = traj_id
        self.data_type = data_type
        self.encoder_type = encoder_type  # "svd" or "sd3"
        self.annotation_dir = annotation_dir
        self.loaded = False

        self._annotation = None
        self._latents = None

        self._load_annotation()

        if load_on_init:
            self.load()

    def _load_annotation(self):
        """Load trajectory annotation file"""
        anno_path = os.path.join(self.data_root, f"{self.annotation_dir}/{self.data_type}/{self.traj_id}.json")
        with open(anno_path, 'r') as f:
            self._annotation = json.load(f)

    def load(self):
        """Load trajectory latent features into memory."""
        if self.loaded:
            return

        if self.encoder_type == "sd3":
            if 'sd3_latent_path' in self._annotation:
                latent_path = os.path.join(self.data_root, self._annotation['sd3_latent_path'])
            else:
                base_path = os.path.join(self.data_root, self._annotation['latent_path'])
                latent_path = base_path.replace('.npy', '_sd3.npz')
        else:
            latent_path = os.path.join(self.data_root, self._annotation['latent_path'])

        if latent_path.endswith('.npz'):
            with np.load(latent_path) as data:
                stacked_latents = data['latents'].astype(np.float32)
        elif os.path.exists(latent_path):
            stacked_latents = np.load(latent_path).astype(np.float32)
        else:
            npz_path = latent_path.replace('.npy', '.npz')
            if os.path.exists(npz_path):
                with np.load(npz_path) as data:
                    stacked_latents = data['latents'].astype(np.float32)
            else:
                raise FileNotFoundError(f"Latent file not found: {latent_path}")

        num_cameras = self._annotation['num_cameras']
        camera_names = CAMERA_NAMES[:num_cameras]
        self._latents = {}
        for i, cam_name in enumerate(camera_names):
            self._latents[cam_name] = stacked_latents[i]

        self.loaded = True

    def get_video_frames(self, start_idx: int, end_idx: int) -> Dict[str, np.ndarray]:
        """Read a specific range of video frames.

        If preprocessed .npy frames exist, uses memory-mapped loading (very fast).
        Otherwise falls back to OpenCV video decoding (slower).

        Args:
            start_idx: Starting frame index (inclusive)
            end_idx: Ending frame index (exclusive)

        Returns:
            Dictionary mapping camera names to numpy arrays of shape (num_frames, H, W, C)
        """
        num_cameras = self._annotation['num_cameras']
        camera_names = CAMERA_NAMES[:num_cameras]

        # Check if preprocessed numpy frames exist (preferred - much faster)
        if 'video_frames_path' in self._annotation:
            npy_path = os.path.join(self.data_root, self._annotation['video_frames_path'])
            if os.path.exists(npy_path):
                frames_mmap = np.load(npy_path, mmap_mode='r')

                result = {}
                for i, cam_name in enumerate(camera_names):
                    result[cam_name] = np.array(frames_mmap[i, start_idx:end_idx])

                return result

        # Fallback to video decoding (slower)
        video_path = os.path.join(self.data_root, self._annotation['video_path'])
        return read_video_frames_range(video_path, start_idx, end_idx, num_cameras)

    def unload(self):
        """Free memory by unloading trajectory data"""
        if self._latents is not None:
            self._latents.clear()
        self._latents = None
        self.loaded = False

    def __len__(self) -> int:
        return self._annotation['video_length']

    @property
    def annotation(self) -> Dict:
        return self._annotation

    @property
    def latents(self) -> Dict[str, torch.Tensor]:
        if not self.loaded:
            self.load()
        return self._latents


class PrecomputedDroid(Dataset):
    """PyTorch dataset for preprocessed DROID trajectories."""

    def __init__(
        self,
        root: str,
        split: str = 'train',
        horizon: int = 16,
        img_keys: List[str] = ['exterior_1_left', 'exterior_2_left', 'wrist_left'],
        relabel_actions: bool = False,
        normalize: bool = True,
        norm_stats_path: Optional[str] = None,
        cache_trajectories: bool = False,
        return_language: bool = True,
        max_trajectories: Optional[int] = None,
        return_video_frames: bool = False,
        encoder_type: str = "svd",
        n_memory_frames: int = 0,
        t_memory: int = 1,
        n_history: int = 2,
        use_fixed_t: bool = False,
        fixed_t: int = 0,
        use_fixed_id: bool = False,
        eval_mode: bool = False,
        reward_key: str = 'reward_progress',
        negative_reward: bool = True,
        annotation_dir: str = 'annotations',
        collapse_prob: float = 0.1,

    ):
        """
        Args:
            root: Root directory of preprocessed DROID dataset
            split: 'train' or 'val'
            horizon: Number of frames to generate (not including history)
            n_history: Number of history/context frames before generation
            img_keys: Which camera views to use (0=exterior_1_left, 1=exterior_2_left, 2=wrist_left)
            relabel_actions: Whether to relabel actions (compute from state differences)
            normalize: Whether to normalize states and actions
            norm_stats_path: Optional normalization-statistics JSON path. Relative
                paths are resolved under the dataset root.
            cache_trajectories: Keep trajectories in memory (high RAM usage)
            return_language: Whether to return language instructions
            max_trajectories: Limit number of trajectories (for debugging)
            return_video_frames: Whether to return raw video frames (useful for saving videos)
            encoder_type: Which encoder features to use ("svd" or "sd3")
        """
        self.root = Path(root)
        self.split = split
        self.horizon = horizon
        self.n_history = n_history
        self.img_keys_list = img_keys
        self.relabel_actions = relabel_actions
        self.normalize = normalize
        self.norm_stats_path = norm_stats_path
        self.cache_trajectories = cache_trajectories
        self.return_language = return_language
        self.return_video_frames = return_video_frames
        self.encoder_type = encoder_type
        self.n_memory_frames = n_memory_frames
        self.t_memory = t_memory
        self.use_fixed_t = use_fixed_t
        self.fixed_t = fixed_t
        self.use_fixed_id = use_fixed_id
        self.eval_mode = eval_mode
        self.reward_key = reward_key
        self.negative_reward = negative_reward
        self.annotation_dir = annotation_dir
        self.collapse_prob = collapse_prob
        self.eps_idx = 0

        transform = "R-1" if self.negative_reward else "raw R"
        print(f"Reward config: key='{self.reward_key}', transform={transform}")

        data_type = 'val' if split == 'valid' else split
        anno_dir = self.root / f"{self.annotation_dir}/{data_type}"

        if not anno_dir.exists():
            raise ValueError(f"Annotation directory not found: {anno_dir}")

        self.traj_ids = []
        for anno_file in sorted(anno_dir.glob("*.json")):
            traj_id = int(anno_file.stem)
            self.traj_ids.append(traj_id)

        if max_trajectories:
            self.traj_ids = self.traj_ids[:max_trajectories]

        print(f"Found {len(self.traj_ids)} {split} trajectories")

        if self.normalize:
            suffix = 'relabel' if relabel_actions else 'recorded'
            if self.norm_stats_path:
                norm_path = Path(self.norm_stats_path).expanduser()
                if not norm_path.is_absolute():
                    norm_path = self.root / norm_path
            else:
                norm_path = self.root / f"norm_stats_{suffix}.json"
            assert norm_path.exists(), (
                f"Normalization is enabled but {norm_path.name} was not found at {norm_path}. "
                "Set dataset.norm_stats_path, add the norm_stats file to the dataset root, "
                "or set dataset.normalize=False."
            )
            with open(norm_path, 'r') as f:
                norm_stats = json.load(f)['norm_stats']
            self.norm_dict = {
                'states': {
                    'mean': torch.tensor(norm_stats['state']['mean']),
                    'std': torch.tensor(norm_stats['state']['std']),
                },
                'actions': {
                    'mean': torch.tensor(norm_stats['actions']['mean']),
                    'std': torch.tensor(norm_stats['actions']['std']),
                }
            }
            print(f"Loaded normalization statistics from {norm_path}")

        self.trajectories: List[DROIDTrajectory] = []
        self.valid_trajectories: List[int] = []

        self._states: List[torch.Tensor] = []
        self._actions: List[torch.Tensor] = []
        self._text_features: List[torch.Tensor] = []
        self._rewards: List[torch.Tensor] = []
        self._texts: List[str] = []

        print("Initializing trajectories and preprocessing states/actions...")
        for traj_id in tqdm(self.traj_ids):
            traj = DROIDTrajectory(
                data_root=str(self.root),
                traj_id=traj_id,
                data_type=data_type,
                load_on_init=cache_trajectories,
                encoder_type=self.encoder_type,
                annotation_dir=self.annotation_dir,
            )

            states, actions, rewards = self._preprocess_states_actions(traj.annotation)
            if (len(traj) >= horizon + 2 and len(states) >= horizon + 2) or self.eval_mode:
                self.trajectories.append(traj)
                self.valid_trajectories.append(len(self.trajectories) - 1)
                self._states.append(states)
                self._actions.append(actions)
                self._rewards.append(rewards)

                if self.return_language:
                    self._texts.append(traj.annotation['texts'][0])
                    if 'text_features' in traj.annotation:
                        self._text_features.append(
                            torch.tensor(traj.annotation['text_features'], dtype=torch.float32)
                        )

        print(f"Loaded {len(self.valid_trajectories)} valid trajectories (length >= {horizon})")

        if len(self.valid_trajectories) == 0:
            raise ValueError(f"No trajectories found with length >= {horizon}")

    def __len__(self) -> int:
        return len(self.valid_trajectories)

    def _preprocess_states_actions(
        self,
        annotation: Dict,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Preprocess full trajectory states and actions from annotation.

        Called once during __init__ to cache preprocessed data.

        Note: The annotation stores raw (full-resolution) states and actions,
        but video/latents are downsampled by rgb_skip. We need to convert
        video frame indices to raw indices using the skip factor.

        Returns:
            states: (T, 8) float32 tensor of states at video frame rate
            actions: (T, 8) float32 tensor of actions at video frame rate
        """
        joint_position = np.array(annotation['observation.state.joint_position'])
        gripper_position = np.array(annotation['observation.state.gripper_position'])[:, None]
        if len(gripper_position.shape) == 3:
            gripper_position = gripper_position[:, 0, :]
        full_states = np.concatenate([joint_position, gripper_position], axis=-1)

        raw_length = len(joint_position)
        state_indices = np.arange(0, raw_length, RGB_SKIP)
        states = full_states[state_indices]
        if self.reward_key not in annotation:
            raise ValueError(
                f"Reward key '{self.reward_key}' not found in annotation for "
                f"episode {annotation.get('episode_id', '<unknown>')}. "
                f"Available keys: {list(annotation.keys())}"
            )
        rewards = np.asarray(annotation[self.reward_key], dtype=np.float32).reshape(-1)
        if self.negative_reward:
            rewards = rewards - 1.0
        if len(rewards) == raw_length:
            rewards = rewards[state_indices]
        elif len(rewards) != len(states):
            raise ValueError(
                f"Reward key '{self.reward_key}' has length {len(rewards)}, but expected "
                f"raw length {raw_length} or downsampled length {len(states)} for "
                f"episode {annotation.get('episode_id', '<unknown>')}."
            )

        if self.relabel_actions:
            next_state_indices = np.clip(state_indices + RGB_SKIP, 0, len(full_states) - 1)
            actions = full_states[next_state_indices] - full_states[state_indices]
        else:
            action_joint = np.array(annotation['action.joint_velocity'])
            action_gripper = np.array(annotation['action.gripper_position'])
            if action_gripper.ndim == 1:
                action_gripper = action_gripper[:, None]

            actions = []
            for i in state_indices:
                end_action_idx = min(i + RGB_SKIP, len(action_joint))
                action_sum = action_joint[i:end_action_idx].sum(axis=0)
                gripper_last = action_gripper[end_action_idx-1]
                full_actions = np.concatenate([action_sum, gripper_last], axis=-1)
                actions.append(full_actions)
            actions = np.stack(actions, axis=0)

        return (
            torch.from_numpy(states).float(),
            torch.from_numpy(actions).float(),
            torch.from_numpy(rewards).float(),
        )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Sample a chunk from a trajectory

        Returns:
            Dict containing:
                - obs: Dict with image latents and states
                  - states: (horizon, 8) float32 joint positions (7D) + gripper position (1D)
                - actions: (horizon, 8) float32 actions (joint deltas + gripper delta)
                - task: language instruction (if return_language=True)
                - rewards: (horizon,) rewards from annotation[dataset.reward_key]
        """
        # Get trajectory index (position in self.trajectories list)
        if self.use_fixed_id:
            idx = self.eps_idx
            self.eps_idx += 1
            if self.eps_idx >= len(self.valid_trajectories):
                self.eps_idx = 0
        traj_idx = self.valid_trajectories[idx]
        traj = self.trajectories[traj_idx]

        # Ensure trajectory latents are loaded
        if not traj.loaded:
            traj.load()

        # Sample t: the last history frame. History = [t-n_history+1, ..., t],
        # generation = [t+1, ..., t+horizon]. Obs chunk = [t-n_history+1, t+horizon].
        n_hist = self.n_history
        
        min_t = n_hist - 1
        max_t = len(traj) - self.horizon - 1
        if self.use_fixed_t:
            t = n_hist - 1 + self.fixed_t
        else:
            t = random.randint(min_t, max_t) if max_t > min_t else min_t

        start_idx = max(0, t - n_hist + 1)
        end_idx = min(len(traj), t + self.horizon + 1)

        # Shared collapse dice: with p=0.1 collapse BOTH the dense history window
        # AND the sparse memory frames so the WM learns to handle degenerate context.
        # History collapse: all history obs -> frame t. History actions at positions
        # [0, n_hist-2] are zeroed in normalized space (full row incl. gripper — the
        # gripper position signal is still present in states[:, -1], so no info loss).
        # Action at n_hist-1 (action at t) is kept intact to drive the first predicted
        # step t+1. Memory collapse: all memory obs -> frame t (effective_t_memory = 0).
        collapse = random.random() < self.collapse_prob
        collapse_history = collapse and n_hist > 1

        # Extract latent features for each camera view
        obs = {}
        for img_key in self.img_keys_list:
            if img_key not in traj.latents:
                raise ValueError(f"Invalid image key: {img_key}. Available cameras: {list(traj.latents.keys())}")

            # Copy the numpy slice so torch.from_numpy doesn't alias the cached
            # trajectory buffer (in-place collapse would otherwise poison the cache
            # when cache_trajectories=True).
            latents = torch.from_numpy(
                np.ascontiguousarray(traj.latents[img_key][start_idx:end_idx])
            )  # (n_hist + horizon, C, H, W)
            if collapse_history:
                latents[:n_hist] = latents[n_hist - 1:n_hist]
            obs[f'{img_key}_features'] = latents

        # Load video frames on-demand using random access (only reads needed frames)
        if self.return_video_frames:
            video_frames = traj.get_video_frames(start_idx, end_idx)
            for img_key in self.img_keys_list:
                if img_key in video_frames:
                    vid = load_and_preprocess_video(video_frames[img_key])
                    if collapse_history:
                        vid[:n_hist] = vid[n_hist - 1:n_hist]
                    obs[img_key] = vid

        # Get pre-loaded states and actions (slice from cached tensors)
        states = self._states[traj_idx][start_idx:end_idx].clone()
        actions = self._actions[traj_idx][start_idx:end_idx].clone()
        rewards = self._rewards[traj_idx][start_idx:end_idx].clone()
        # Normalize if enabled
        if self.normalize:
            states = (states - self.norm_dict['states']['mean']) / self.norm_dict['states']['std']
            actions = (actions - self.norm_dict['actions']['mean']) / self.norm_dict['actions']['std']

        if collapse_history:
            # Replicate state at position n_hist-1 across the history window
            states[:n_hist] = states[n_hist - 1:n_hist]
            # Zero history actions at positions [0, n_hist-2]; keep action at n_hist-1
            # (drives prediction of t+1). Gripper info is retained via states[:, -1].
            actions[:n_hist - 1] = 0.0

        obs['states'] = states
        

        result = {
            'obs': obs,
            'actions': actions,
            'rewards': rewards,
        }

        # Build sparse memory frames before history, spaced by t_memory from t.
        # Memory collapse is coupled to the same `collapse` dice as history collapse:
        # when collapse fires (p=0.1), effective_t_memory = 0 so all memory frames
        # repeat the current frame t (matches Ctrl-World dataset_droid_exp33.py:152-160
        # where p < 0.15 sets skip_his = 0).
        if self.n_memory_frames > 0:
            if collapse:
                effective_t_memory = 0
            else:
                effective_t_memory = self.t_memory
            # Clamp to valid range for both latents and states (states_len can be shorter than traj_len)
            memory_indices = [
                max(0, t - (self.n_memory_frames - i) * effective_t_memory)
                for i in range(self.n_memory_frames)
            ]

            memory_obs = {}

            # Latent features for each camera. np.stack already produces a fresh
            # array, so no aliasing concern here — but we still wrap in ascontiguous
            # to be consistent with the main obs path.
            for img_key in self.img_keys_list:
                mem_latents = np.stack([traj.latents[img_key][mi] for mi in memory_indices])
                memory_obs[f'{img_key}_features'] = torch.from_numpy(np.ascontiguousarray(mem_latents))

            # Raw video frames if requested
            if self.return_video_frames:
                for img_key in self.img_keys_list:
                    mem_frames = []
                    for mi in memory_indices:
                        frame_dict = traj.get_video_frames(mi, mi + 1)
                        if img_key in frame_dict:
                            mem_frames.append(frame_dict[img_key][0])
                    if mem_frames:
                        memory_obs[img_key] = load_and_preprocess_video(np.stack(mem_frames))

            # States at memory indices
            mem_states = torch.stack([self._states[traj_idx][mi] for mi in memory_indices])
            if self.normalize:
                mem_states = (mem_states - self.norm_dict['states']['mean']) / self.norm_dict['states']['std']
            memory_obs['states'] = mem_states

            result['memory'] = memory_obs

        if self.return_language:
            result['task'] = {
                'text': self._texts[traj_idx],
                'features': self._text_features[traj_idx].clone(),
            }

        # Unload latents if not caching (states/actions remain cached)
        if not self.cache_trajectories:
            traj.unload()

        return result


def load_and_preprocess_video(images: np.ndarray) -> torch.Tensor:
    """Preprocess video frames: normalize and rearrange to (T, C, H, W)"""
    images = images.astype(np.float32) / 255.0
    images = rearrange(images, 't h w c -> t c h w')
    return torch.tensor(images).contiguous()
