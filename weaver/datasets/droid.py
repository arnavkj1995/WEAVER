"""
DROID PyTorch Dataset for World Models
Loads preprocessed DROID data with latent features
Samples trajectory chunks of fixed horizon for world model training
"""

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
    camera_names = ['exterior_1_left', 'exterior_2_left', 'wrist_left'][:num_cameras]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file {video_path}")

    # Seek to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

    # Get video dimensions
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    camera_width = frame_width // num_cameras

    # Read only the frames we need
    frames_by_camera = {cam: [] for cam in camera_names}

    for _ in range(end_idx - start_idx):
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Split frame by camera
        for i, cam_name in enumerate(camera_names):
            start_w = i * camera_width
            end_w = (i + 1) * camera_width
            frames_by_camera[cam_name].append(frame[:, start_w:end_w, :])

    cap.release()

    # Stack frames into arrays
    result = {}
    for cam_name, frames in frames_by_camera.items():
        if frames:
            result[cam_name] = np.stack(frames, axis=0)

    return result


class DROIDTrajectory:
    """Container for a single DROID trajectory"""

    def __init__(self, data_root: str, traj_id: int, data_type: str, load_on_init: bool = False, encoder_type: str = "svd", annotation_dir: str = "annotation_rewards"):
        self.data_root = data_root
        self.traj_id = traj_id
        self.data_type = data_type
        self.encoder_type = encoder_type  # "svd" or "sd3"
        self.annotation_dir = annotation_dir
        self.loaded = False

        self._annotation = None
        self._video = None
        self._latents = None

        # Load annotation to get metadata
        self._load_annotation()

        if load_on_init:
            self.load()

    def _load_annotation(self):
        """Load trajectory annotation file"""
        anno_path = os.path.join(self.data_root, f"{self.annotation_dir}/{self.data_type}/{self.traj_id}.json")
        with open(anno_path, 'r') as f:
            self._annotation = json.load(f)

    def load(self, _load_video: bool = False):
        """Load trajectory latent features into memory.

        Args:
            _load_video: Deprecated, ignored. Use get_video_frames() for video access.
        """
        if self.loaded:
            return

        # # Skip trajectories with 2 cameras (inconsistent with 3-camera setup)
        # if self._annotation.get('num_cameras') == 2:
        #     return

        # Load latent features as numpy (better memory handling with DataLoader workers)
        # Shape: [num_cameras, num_frames, C, H, W]
        latent_path = os.path.join(self.data_root, self._annotation['latent_path'])
    
        # Modify path based on encoder type (sd3 uses compressed npz format)
        if self.encoder_type == "sd3":
            latent_path = latent_path.replace('.npy', '_sd3.npz')

        # Support .npy, .npz (compressed), and .pt formats
        if latent_path.endswith('.npz'):
            with np.load(latent_path) as data:
                stacked_latents = data['latents'].astype(np.float32)
        elif os.path.exists(latent_path):
            # Load from uncompressed npy format (auto-converts fp16 to fp32)
            stacked_latents = np.load(latent_path).astype(np.float32)
        else:
            npz_path = latent_path.replace('.npy', '.npz')
            if os.path.exists(npz_path):
                with np.load(npz_path) as data:
                    stacked_latents = data['latents'].astype(np.float32)
            else:
                raise FileNotFoundError(f"Latent file not found: {latent_path}")
        
      
        # Split latents by camera - numpy slicing creates lightweight views
        num_cameras = self._annotation['num_cameras']
        camera_names = ['exterior_1_left', 'exterior_2_left', 'wrist_left'][:num_cameras]
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
        camera_names = ['exterior_1_left', 'exterior_2_left', 'wrist_left'][:num_cameras]

        # Check if preprocessed numpy frames exist (preferred - much faster)
        if 'video_frames_path' in self._annotation:
            npy_path = os.path.join(self.data_root, self._annotation['video_frames_path'])
            if os.path.exists(npy_path):
                # Memory-mapped loading - only reads the requested frames from disk
                # Shape: (num_cameras, T, H, W, C)
                frames_mmap = np.load(npy_path, mmap_mode='r')

                result = {}
                for i, cam_name in enumerate(camera_names):
                    # Slicing a mmap array only reads those bytes from disk
                    # Need to copy to avoid issues with mmap in DataLoader workers
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
    """
    PyTorch Dataset for preprocessed DROID data

    Sampling strategy:
    1. Sample a trajectory
    2. Sample a random chunk of length `horizon` from that trajectory
    """

    def __init__(
        self,
        root: str,
        split: str = 'train',
        horizon: int = 16,
        img_keys: List[str] = ['exterior_1_left', 'exterior_2_left', 'wrist_left'],
        relabel_actions: bool = False,
        normalize: bool = True,
        cache_trajectories: bool = False,
        return_language: bool = True,
        load_precomputed_features: bool = False,
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
        annotation_dir: str = 'annotation_rewards',
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
            cache_trajectories: Keep trajectories in memory (high RAM usage)
            return_language: Whether to return language instructions
            load_precomputed_features: Whether to load precomputed text features
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
        self.cache_trajectories = cache_trajectories
        self.return_language = return_language
        self.load_precomputed_features = load_precomputed_features
        self.return_video_frames = return_video_frames
        self.encoder_type = encoder_type
        self.n_memory_frames = n_memory_frames
        self.t_memory = t_memory
        self.use_fixed_t = use_fixed_t
        self.fixed_t = fixed_t
        self.use_fixed_id = use_fixed_id
        self.eval_mode = eval_mode
        self.annotation_dir = annotation_dir
        self.collapse_prob = collapse_prob
        self.eps_idx = 0
       
        # Map string keys to video indices
        self.camera_map = {
            'exterior_1_left': 0,
            'exterior_2_left': 1,
            'wrist_left': 2
        }

        # Load trajectory list from annotations
        data_type = 'val' if split == 'valid' else split
        anno_dir = self.root / f"{self.annotation_dir}/{data_type}"

        if not anno_dir.exists():
            raise ValueError(f"Annotation directory not found: {anno_dir}")

        # Get all trajectory IDs from annotation files
        self.traj_ids = []
        for anno_file in sorted(anno_dir.glob("*.json")):
            traj_id = int(anno_file.stem)
            self.traj_ids.append(traj_id)

        if max_trajectories:
            self.traj_ids = self.traj_ids[:max_trajectories]

        print(f"Found {len(self.traj_ids)} {split} trajectories")

        # Load normalization statistics (filename depends on relabel_actions)
        if self.normalize:
            suffix = 'relabel' if relabel_actions else 'recorded'
            norm_path = self.root / f"norm_stats_{suffix}.json"
            assert norm_path.exists(), (
                f"Normalization is enabled but {norm_path.name} was not found at {norm_path}. "
                "Either add the norm_stats file or set dataset.normalize=False."
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

        # Create trajectory objects and preprocess states/actions
        self.trajectories: List[DROIDTrajectory] = []
        self.valid_trajectories: List[int] = []

        # Pre-loaded data for fast access (indexed by position in self.trajectories)
        self._states: List[torch.Tensor] = []      # Each: (T, 8) float32
        self._actions: List[torch.Tensor] = []     # Each: (T, 8) float32
        self._text_features: List[torch.Tensor] = []  # Each: (feat_dim,) float32
        self._rewards: List[torch.Tensor] = [] # Each: (T, 1) float32
        self._texts: List[str] = []                # Language instructions

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
            # # Skip trajectories with 2 cameras (inconsistent with 3-camera setup)
            # if traj.annotation.get('num_cameras') == 2:
            #     continue
            # Preprocess and cache states/actions for this trajectory
            states, actions,rewards = self._preprocess_states_actions(traj.annotation)
             # Require both video and states to be long enough (lengths can differ)
            # if not self.pad_short_trajectories:
            if (len(traj) >= horizon + 2 and len(states) >= horizon + 2) or self.eval_mode:
                self.trajectories.append(traj)
                self.valid_trajectories.append(len(self.trajectories) - 1)
                self._states.append(states)
                self._actions.append(actions)
                self._rewards.append(rewards)
            # else: 
            #     target_length = horizon + 2
            #     # Pad short trajectories by repeating last frame
            #     if len(traj) < target_length or len(states) < target_length:
            #         traj.load()  # Need latents for padding
            #         states, actions, padded_latents = pad_trajectory_to_length(
            #             states, actions, traj._latents, target_length
            #         )
            #         traj._latents = padded_latents
            #         traj._annotation['video_length'] = target_length
            #         print(f"Padded trajectory {traj_id} to length {target_length}")
            #     # Add trajectory (either original or padded)
            #     self.trajectories.append(traj)
            #     self.valid_trajectories.append(len(self.trajectories) - 1)
            #     self._states.append(states)
            #     self._actions.append(actions)

                # Cache text features and text
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Preprocess full trajectory states and actions from annotation.

        Called once during __init__ to cache preprocessed data.

        Note: The annotation stores raw (full-resolution) states and actions,
        but video/latents are downsampled by rgb_skip. We need to convert
        video frame indices to raw indices using the skip factor.

        Returns:
            states: (T, 8) float32 tensor of states at video frame rate
            actions: (T, 8) float32 tensor of actions at video frame rate
        """
        rgb_skip = 3  # Hardcoded for now

        # Get joint states (joint_position + gripper_position) at raw resolution
        joint_position = np.array(annotation['observation.state.joint_position'])
        
        gripper_position = np.array(annotation['observation.state.gripper_position'])[:, None]
        if len(gripper_position.shape) == 3:
            gripper_position = gripper_position[:, 0, :]
        full_states = np.concatenate([joint_position, gripper_position], axis=-1)

        # Sample states at video frame rate (every rgb_skip frames)
        raw_length = len(joint_position)
        state_indices = np.arange(0, raw_length, rgb_skip)
        states = full_states[state_indices]
        rewards = np.zeros(len(states), dtype=np.float32)

        if self.relabel_actions:
            # Compute actions as state differences between consecutive video frames
            next_state_indices = np.clip(state_indices + rgb_skip, 0, len(full_states) - 1)
            actions = full_states[next_state_indices] - full_states[state_indices]
        else:
            # Use recorded actions - sum actions between video frames
            action_joint = np.array(annotation['action.joint_velocity'])
            action_gripper = np.array(annotation['action.gripper_position'])
            if action_gripper.ndim == 1:
                action_gripper = action_gripper[:, None]
            #full_actions = np.concatenate([action_joint, action_gripper], axis=-1)

            # Sum actions over each rgb_skip interval to get action per video frame
            actions = []
            for i in state_indices:
                end_action_idx = min(i + rgb_skip, len(action_joint))
                action_sum = action_joint[i:end_action_idx].sum(axis=0)
                gripper_last = action_gripper[end_action_idx-1]
                full_actions = np.concatenate([action_sum, gripper_last],axis=-1)
                actions.append(full_actions)
            actions = np.stack(actions, axis=0)

        return (
            torch.from_numpy(states).float(),
            torch.from_numpy(actions).float(),
            torch.from_numpy(rewards).float() if rewards is not None else None
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
                - rewards: (horizon,) rewards (zeros for DROID)
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
        ## add an option here to use the random t or a fixed t for the generation
        if self.use_fixed_t:
            t = n_hist -1 + self.fixed_t
        else:
            t = random.randint(min_t, max_t) if max_t > min_t else min_t
        
        ## get the max of 0 and t - n_hist + 1
        start_idx = max(0, t - n_hist + 1)
        # start_idx = t - n_hist + 1
        # end_idx = t + self.horizon + 1
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
