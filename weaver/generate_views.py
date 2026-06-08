"""Generate WEAVER rollouts and save per-camera prediction views.

This is the standalone evaluation/data-generation entrypoint. It intentionally
does not compute metrics; use scripts/compute_metrics.py on the saved views.
"""

from __future__ import annotations

import argparse
import gc
import os
import time
from pathlib import Path

import imageio
import numpy as np
import torch
import yaml

from .datasets.droid import PrecomputedDroid
from .utils.config import dict_to_namespace, load_config, merge_dicts, update_config
from .utils.tools import load_checkpoint
from .wm.encoders import get_encoder, get_task_encoder
from .wm.model import WEAVER


def load_eval_config(checkpoint_dir: str, overrides: list[str], config_path: str | None = None):
    default_cfg = load_config(Path(__file__).with_name("config.yaml"), mode="defaults")
    config_path = Path(config_path) if config_path else Path(checkpoint_dir) / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with config_path.open("r") as f:
        cfg_dict = yaml.safe_load(f)
    if "defaults" in cfg_dict and isinstance(cfg_dict["defaults"], dict):
        cfg_dict = cfg_dict["defaults"]
    cfg_dict = merge_dicts(default_cfg, cfg_dict)

    if overrides:
        cfg_dict = update_config(cfg_dict, dict(item.split("=", 1) for item in overrides))
    return dict_to_namespace(cfg_dict)


def image_keys_from_cfg(cfg) -> list[str]:
    if cfg.dataset.sample_aux_from_left_right:
        return ["aux", "wrist"]
    return list(cfg.dataset.img_keys)


def build_model(cfg, device: str):
    img_keys = image_keys_from_cfg(cfg)
    image_size = cfg.dataset.image_size
    if isinstance(image_size, int):
        image_size = (image_size, image_size)

    im_encoder, train_decoder = get_encoder(
        config=cfg.im_encoder,
        image_size=cfg.dataset.image_size,
        device=device,
    )
    task_encoder = get_task_encoder(config=None, device=device)

    model = WEAVER(
        img_keys=img_keys,
        im_encoder=im_encoder,
        train_decoder=train_decoder,
        task_encoder=task_encoder,
        n_history=cfg.n_history,
        n_horizon=cfg.horizon,
        config=cfg.model,
        use_precomputed_features=False,
        n_states=cfg.dataset.n_states,
        n_actions=cfg.dataset.n_actions,
        image_size=image_size,
        device=device,
        n_memory_frames=cfg.n_memory_frames,
        t_memory=cfg.t_memory,
        inference_config=cfg.inference,
    ).to(device)
    model.ema.to(device)
    return model, img_keys


def clean_state_dict(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        if key.startswith("_orig_mod."):
            key = key[len("_orig_mod."):]
        cleaned[key] = value
    return cleaned


def load_trajectories(cfg, split: str, img_keys: list[str], max_trajectories: int):
    encoder_name = cfg.im_encoder.name
    encoder_type = "sd3" if "stable-diffusion-3" in encoder_name or "sd3" in encoder_name else "svd"

    dataset = PrecomputedDroid(
        root=cfg.dataset.path,
        split=split,
        horizon=cfg.eval_video_frames,
        img_keys=img_keys,
        relabel_actions=cfg.dataset.relabel_actions,
        normalize=cfg.dataset.normalize,
        cache_trajectories=False,
        return_language=True,
        max_trajectories=max_trajectories,
        return_video_frames=True,
        encoder_type=encoder_type,
        n_memory_frames=cfg.n_memory_frames,
        t_memory=cfg.t_memory,
        n_history=cfg.n_history,
        eval_mode=True,
        annotation_dir=cfg.dataset.annotation_dir,
    )

    trajectories = []
    for traj_idx in dataset.valid_trajectories[:max_trajectories]:
        traj = dataset.trajectories[traj_idx]
        text_features = dataset._text_features[traj_idx] if len(dataset._text_features) > traj_idx else None
        trajectories.append(
            {
                "traj": traj,
                "traj_id": traj.traj_id,
                "states": dataset._states[traj_idx],
                "actions": dataset._actions[traj_idx],
                "text": dataset._texts[traj_idx] if len(dataset._texts) > traj_idx else "",
                "text_features": text_features,
                "norm_dict": dataset.norm_dict if dataset.normalize else None,
                "length": len(traj),
            }
        )
    return trajectories


def gather_generation_inputs(traj_info, cfg, img_keys, start_idx: int, use_real_history: bool, device: str):
    traj = traj_info["traj"]
    states = traj_info["states"]
    actions = traj_info["actions"]
    norm_dict = traj_info["norm_dict"]
    traj_len = traj_info["length"]

    start_idx = max(0, min(start_idx, traj_len - 1))
    if start_idx >= traj_len - 1:
        return None

    n_hist = cfg.n_history
    use_real = use_real_history and start_idx > 0
    if use_real:
        hist_idx = np.clip(np.arange(start_idx - n_hist + 1, start_idx + 1), 0, traj_len - 1)
    else:
        hist_idx = np.full(n_hist, start_idx, dtype=np.int64)
    roll_idx = np.arange(start_idx + 1, traj_len)
    if len(roll_idx) == 0:
        return None

    horizon = cfg.eval_horizon
    bootstrap = cfg.eval_bootstrap or horizon
    n_pad = ((-len(roll_idx)) % bootstrap) + max(0, horizon - bootstrap)
    roll_idx_padded = (
        np.concatenate([roll_idx, np.full(n_pad, traj_len - 1, dtype=np.int64)])
        if n_pad
        else roll_idx
    )
    obs_idx = np.concatenate([hist_idx, roll_idx_padded])

    n_memory = cfg.n_memory_frames
    t_memory = cfg.t_memory
    if n_memory > 0:
        if use_real:
            mem_idx = np.clip(np.arange(start_idx - n_memory * t_memory, start_idx, t_memory), 0, traj_len - 1)
        else:
            mem_idx = np.full(n_memory, start_idx, dtype=np.int64)
    else:
        mem_idx = None

    all_idx = obs_idx if mem_idx is None else np.concatenate([obs_idx, mem_idx])
    load_start = int(all_idx.min())
    load_end = int(all_idx.max()) + 1
    video_frames_full = traj.get_video_frames(load_start, load_end)
    if any(key not in video_frames_full for key in img_keys):
        return None

    def gather_frames(key, idx):
        arr = video_frames_full[key][idx - load_start]
        tensor = torch.from_numpy(arr).float().permute(0, 3, 1, 2) / 255.0
        return tensor.unsqueeze(0).to(device)

    def gather_states(idx):
        tensor = states[idx].clone()
        if norm_dict:
            tensor = (tensor - norm_dict["states"]["mean"]) / norm_dict["states"]["std"]
        return tensor.unsqueeze(0).to(device)

    obs = {key: gather_frames(key, obs_idx) for key in img_keys}
    obs["states"] = gather_states(obs_idx)

    action_slice = actions[obs_idx].clone()
    if norm_dict:
        action_slice = (action_slice - norm_dict["actions"]["mean"]) / norm_dict["actions"]["std"]
    action_batch = action_slice.unsqueeze(0).to(device)
    if n_hist > 1:
        action_batch[:, : n_hist - 1] = 0.0

    memory = None
    if mem_idx is not None:
        memory = {key: gather_frames(key, mem_idx) for key in img_keys}
        memory["states"] = gather_states(mem_idx)

    gt_views = {key: video_frames_full[key][roll_idx - load_start] for key in img_keys}
    task = {"text": traj_info["text"]}
    if traj_info["text_features"] is not None:
        task["features"] = traj_info["text_features"].unsqueeze(0).to(device)

    return obs, action_batch, memory, task, gt_views, roll_idx


@torch.no_grad()
def generate_and_save(model, cfg, img_keys, args, device: str):
    trajectories = load_trajectories(cfg, args.split, img_keys, args.num_samples)
    if args.val_chunk > 1:
        splits = np.array_split(np.arange(len(trajectories)), args.val_chunk)
        keep = splits[args.val_chunk_id].tolist()
        trajectories = [trajectories[i] for i in keep]
        print(f"Chunk {args.val_chunk_id}/{args.val_chunk}: {len(trajectories)} trajectories")

    views_dir = Path(args.output_dir) / "views"
    videos_dir = Path(args.output_dir) / "videos"
    views_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    saved_videos = 0
    for traj_info in trajectories:
        traj = traj_info["traj"]
        traj_id = traj_info["traj_id"]
        if not traj.loaded:
            traj.load()
        try:
            inputs = gather_generation_inputs(
                traj_info, cfg, img_keys, args.start_idx, args.use_real_history, device
            )
            if inputs is None:
                print(f"Skipping trajectory {traj_id}: too short or missing camera")
                continue
            obs, actions, memory, task, gt_views, roll_idx = inputs

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                gen_start = time.perf_counter()
                _, decoded = model.generate_videos_full(
                    obs=obs,
                    actions=actions,
                    instructions=task,
                    horizon=cfg.eval_horizon,
                    memory=memory,
                    bootstrap=cfg.eval_bootstrap,
                    use_kv_cache=args.use_kv_cache,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                gen_time = time.perf_counter() - gen_start
                if args.use_kv_cache:
                    print(f"KV_CACHE_GENERATION_TIME traj_id={traj_id} seconds={gen_time:.4f}", flush=True)

            n_hist = cfg.n_history
            t_out = min(decoded[img_keys[0]].shape[1] - n_hist, len(roll_idx))
            pred_views = {}
            for key in img_keys:
                pred_views[key] = (
                    decoded[key][0, n_hist : n_hist + t_out]
                    .float()
                    .cpu()
                    .permute(0, 2, 3, 1)
                    .clamp(0, 1)
                    .mul(255)
                    .byte()
                    .numpy()
                )

            traj_dir = views_dir / str(traj_id)
            traj_dir.mkdir(parents=True, exist_ok=True)
            for vi, key in enumerate(img_keys):
                gt_view = gt_views[key][:t_out]
                if len(gt_view) != len(pred_views[key]):
                    raise ValueError(
                        f"GT/prediction length mismatch for trajectory {traj_id}, "
                        f"{key}: {len(gt_view)} != {len(pred_views[key])}"
                    )
                np.save(traj_dir / f"gt_view{vi}.npy", gt_view)
                np.save(traj_dir / f"pred_view{vi}.npy", pred_views[key])

            if saved_videos < args.num_videos:
                gt = np.concatenate([gt_views[key][:t_out] for key in img_keys], axis=2)
                pred = np.concatenate([pred_views[key] for key in img_keys], axis=2)
                video = np.concatenate([gt, pred], axis=1)
                imageio.mimwrite(videos_dir / f"eval_sample_{traj_id}.mp4", video, fps=5, codec="libx264", format="FFMPEG", quality=8)
                saved_videos += 1

            saved += 1
            del decoded, obs, actions, memory, task
            torch.cuda.empty_cache()
            gc.collect()
        finally:
            traj.unload()

    print(f"Saved {saved} trajectories to {views_dir}")
    return saved


def parse_args():
    parser = argparse.ArgumentParser(description="Generate WEAVER saved views")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--num-samples", type=int, default=120)
    parser.add_argument("--num-videos", type=int, default=8)
    parser.add_argument("--start-idx", type=int, default=20)
    parser.add_argument("--use-real-history", action="store_true")
    parser.set_defaults(use_ema=True)
    parser.add_argument("--use-ema", dest="use_ema", action="store_true")
    parser.add_argument("--no-use-ema", dest="use_ema", action="store_false")
    parser.add_argument("--use-kv-cache", action="store_true")
    parser.add_argument("--val-chunk", type=int, default=1)
    parser.add_argument("--val-chunk-id", type=int, default=0)
    parser.add_argument("--overrides", nargs="*", default=[])
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_dir = os.path.abspath(args.checkpoint)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = load_eval_config(checkpoint_dir, args.overrides, args.config)
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"Output: {output_dir}")
    print(f"Device: {device}")

    model, img_keys = build_model(cfg, device)
    ckpt = load_checkpoint(checkpoint_dir, device, weights_only=True)
    missing, unexpected = model.load_state_dict(clean_state_dict(ckpt["model"]), strict=False)
    print(f"Loaded model with {len(missing)} missing and {len(unexpected)} unexpected keys")
    if missing:
        print("Missing keys:", missing[:20])
    if unexpected:
        print("Unexpected keys:", unexpected[:20])

    if args.use_ema:
        print("Using EMA weights")
        model.ema.load_state_dict(ckpt["ema"])
        model.ema.apply_to(model)

    model.eval()
    start = time.time()
    count = generate_and_save(model, cfg, img_keys, args, device)
    print(f"Done: {count} trajectories in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
