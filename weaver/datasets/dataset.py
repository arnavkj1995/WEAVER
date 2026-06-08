import torch

from typing import Optional

import torch.distributed as dist
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, DistributedSampler


def create_dataloader(
    dataset: Dataset,
    use_ddp: bool = False,
    B: int = 32,
    n_workers: int = 4
) -> DataLoader:

    if use_ddp:
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),  # total GPUs/processes
            rank=dist.get_rank(),                # current process id
            shuffle=True,
            drop_last=True,  # Avoid incomplete batches that break torch.compile
        )

        loader = DataLoader(
            dataset,
            batch_size=B,
            sampler=sampler,
            num_workers=n_workers,
            pin_memory=True,
            pin_memory_device="cuda",
            persistent_workers=True,
            prefetch_factor=4,
            drop_last=True,  # Avoid incomplete batches that break torch.compile
        )

        return loader

    else:
        loader = DataLoader(
            dataset,
            batch_size=B,
            shuffle=True,
            num_workers=n_workers,
            pin_memory=True,
            pin_memory_device="cuda",
            persistent_workers=True,
            prefetch_factor=4,
            drop_last=True,  # Avoid incomplete batches that break torch.compile
        )

        return loader


def create_dataset(
    cfg: dict,
    horizon: int,
    ddp: bool,
    batch_size: int,
    n_workers: int,
    split: str,
    max_trajectories: Optional[int] = None,
    return_video_frames: bool = False,
    im_encoder_name: Optional[str] = None,
    n_memory_frames: int = 0,
    t_memory: int = 1,
    n_history: int = 2,
    use_fixed_t: bool = False,
    fixed_t: int = 0,
    use_fixed_id: bool = False,
    eval_mode: bool = False,
) -> DataLoader:
    # Derive encoder_type from im_encoder name
    encoder_type = 'svd'  # default
    if im_encoder_name is not None:
        if 'stable-diffusion-3' in im_encoder_name:
            encoder_type = 'sd3'
        elif 'stable-video-diffusion' in im_encoder_name or 'stable-diffusion-2' in im_encoder_name:
            encoder_type = 'svd'

    if cfg.name != 'DROID':
        raise ValueError(f"Unsupported dataset '{cfg.name}'. This WEAVER release includes DROID only.")

    from .droid import PrecomputedDroid
    dataset = PrecomputedDroid(
        root=cfg.path,
        split=split,
        horizon=horizon,
        img_keys=cfg.img_keys if hasattr(cfg, 'img_keys') else ['exterior_1_left', 'wrist_left'],
        relabel_actions=cfg.relabel_actions if hasattr(cfg, 'relabel_actions') else False,
        normalize=cfg.normalize if hasattr(cfg, 'normalize') else True,
        norm_stats_path=cfg.norm_stats_path if hasattr(cfg, 'norm_stats_path') else None,
        cache_trajectories=False,
        return_language=True,
        max_trajectories=max_trajectories,
        return_video_frames=return_video_frames,
        encoder_type=encoder_type,
        n_memory_frames=n_memory_frames,
        t_memory=t_memory,
        n_history=n_history,
        use_fixed_t=use_fixed_t,
        fixed_t=fixed_t,
        use_fixed_id=use_fixed_id,
        eval_mode=eval_mode,
        reward_key=cfg.reward_key if hasattr(cfg, 'reward_key') else 'reward_progress',
        negative_reward=cfg.negative_reward if hasattr(cfg, 'negative_reward') else True,
        annotation_dir=cfg.annotation_dir if hasattr(cfg, 'annotation_dir') else 'annotations',
        collapse_prob=cfg.collapse_prob if hasattr(cfg, 'collapse_prob') else 0.1,
    )

    return create_dataloader(
        dataset,
        use_ddp=ddp,
        B=batch_size,
        n_workers=n_workers,
    )
