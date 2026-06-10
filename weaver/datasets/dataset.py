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


def passes_annotation_filter(
    anno: dict,
    filter_episode_id: Optional[str],
    filter_success: bool,
) -> bool:
    """Return True if annotation passes both optional filters.

    Args:
        anno:              Trajectory annotation dict.
        filter_episode_id: Keep only trajectories whose episode_id_orig contains
                           this substring. None = no filter.
        filter_success:    If True, keep only trajectories with success == 1.
    """
    if filter_episode_id:
        ep_id = anno.get("episode_id_orig", str(anno.get("episode_id", "")))
        if filter_episode_id not in str(ep_id):
            return False
    return not (filter_success and anno.get("success", 0) != 1)


def create_synth_dataloader(
    cfg,
    n_history: int,
    horizon: int,
    batch_size: int,
    n_workers: int,
    encoder_name: str,
    filter_episode_id: Optional[str] = None,
    filter_success: bool = False,
) -> DataLoader:
    """Build a DataLoader for synthetic trajectory generation.

    Creates a PrecomputedDroid dataset with video frames enabled (needed for the
    first PI query) and no collapse augmentation, then optionally narrows it by
    episode_id substring or success flag.

    Args:
        cfg:               Dataset config namespace (cfg.dataset from load_eval_config).
        n_history:         Number of history frames to include per sample.
        horizon:           Number of future frames per sample.
        batch_size:        DataLoader batch size (B).
        n_workers:         DataLoader worker count.
        encoder_name:      im_encoder name string; used to pick svd vs sd3 latents.
        filter_episode_id: Optional substring filter on annotation episode_id_orig.
        filter_success:    If True, keep only successful trajectories.

    Returns:
        DataLoader yielding batches with obs, actions, rewards, and task fields.
    """
    encoder_type = "sd3" if ("stable-diffusion-3" in encoder_name or "sd3" in encoder_name) else "svd"
    img_keys = list(cfg.img_keys) if hasattr(cfg, "img_keys") else ["exterior_1_left", "wrist_left"]

    from .droid import PrecomputedDroid
    dataset = PrecomputedDroid(
        root=cfg.path,
        split="train",
        horizon=horizon,
        img_keys=img_keys,
        relabel_actions=getattr(cfg, "relabel_actions", False),
        normalize=getattr(cfg, "normalize", True),
        norm_stats_path=getattr(cfg, "norm_stats_path", None),
        cache_trajectories=False,
        return_language=True,
        return_video_frames=True,
        encoder_type=encoder_type,
        n_memory_frames=0,
        t_memory=1,
        n_history=n_history,
        reward_key=getattr(cfg, "reward_key", "reward_progress"),
        negative_reward=getattr(cfg, "negative_reward", True),
        annotation_dir=getattr(cfg, "annotation_dir", "annotations"),
        collapse_prob=0.0,
    )

    if filter_episode_id or filter_success:
        keep = [
            vi for vi, ti in enumerate(dataset.valid_trajectories)
            if passes_annotation_filter(
                dataset.trajectories[ti].annotation, filter_episode_id, filter_success
            )
        ]
        dataset.valid_trajectories = [dataset.valid_trajectories[i] for i in keep]
        print(f"After filtering: {len(dataset.valid_trajectories)} valid trajectories")

    return create_dataloader(dataset, use_ddp=False, B=batch_size, n_workers=n_workers)


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
