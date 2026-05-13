"""Finetune WEAVER from a pretrained checkpoint.

This script intentionally keeps only the WEAVER finetuning path: load a
pretrained checkpoint, optionally resume an existing finetune checkpoint, train on
``cfg.dataset``, and save to ``logs/chkpts_<suffix>``. Architecture fields are
inherited from the pretrained run so the finetune command only needs data and
training overrides.
"""

import argparse
import copy
import os
import signal
import sys
import time
from datetime import timedelta
from pathlib import Path

import imageio
import numpy as np
import torch
import torch.distributed as dist
import yaml
from einops import rearrange
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from .utils.config import dict_to_namespace, merge_dicts, parse_config, update_config
from .wm.encoders import get_encoder, get_task_encoder
from .wm.model import WEAVER
from .wm.nets import STAttentionBlock
from .datasets import create_dataset
from .utils.tools import cycle, get_lr, load_checkpoint, move_tensors_to_device, save_checkpoint
from .utils.eval import evaluate_with_metrics_ddp


torch.set_float32_matmul_precision("high")


PRETRAINED_INHERIT_KEYS = [
    "model",
    "im_encoder",
    "n_history",
    "horizon",
    "n_memory_frames",
    "t_memory",
    "eval_horizon",
    "eval_video_frames",
    "eval_bootstrap",
    "inference",
]


def load_pretrained_config(pretrained_dir: str) -> dict | None:
    config_path = Path(pretrained_dir) / "config.yaml"
    if not config_path.exists():
        print(f"WARNING: no pretrained config found at {config_path}")
        return None
    with config_path.open("r") as f:
        return yaml.safe_load(f)


def inherit_pretrained_config(cfg_dict: dict, pretrained_cfg: dict | None) -> dict:
    if pretrained_cfg is None:
        return cfg_dict

    cfg_dict = copy.deepcopy(cfg_dict)
    for key in PRETRAINED_INHERIT_KEYS:
        if key in pretrained_cfg:
            if isinstance(cfg_dict.get(key), dict) and isinstance(pretrained_cfg[key], dict):
                cfg_dict[key] = merge_dicts(copy.deepcopy(cfg_dict[key]), copy.deepcopy(pretrained_cfg[key]))
            else:
                cfg_dict[key] = pretrained_cfg[key]

    pretrained_dataset = pretrained_cfg.get("dataset", {})
    cfg_dict.setdefault("dataset", {})
    for key in ["img_keys", "image_size", "n_states", "n_actions"]:
        if key in pretrained_dataset:
            cfg_dict["dataset"][key] = pretrained_dataset[key]

    return cfg_dict


def apply_finetune_config(cfg_dict: dict) -> dict:
    cfg_dict = copy.deepcopy(cfg_dict)
    finetune_cfg = cfg_dict.get("finetune_cfg", {})
    if finetune_cfg:
        cfg_dict.setdefault("training", {})
        cfg_dict["training"].update(finetune_cfg)
        print(f"Applied finetune_cfg to training: {finetune_cfg}")
    return cfg_dict


def setup_ddp() -> tuple[bool, int, int, int, bool]:
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        init_process_group(backend="nccl", timeout=timedelta(hours=2))
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(f"cuda:{local_rank}")
        return True, rank, local_rank, world_size, rank == 0
    return False, 0, 0, 1, True


def image_keys_from_cfg(cfg) -> list[str]:
    if cfg.dataset.sample_aux_from_left_right:
        return ["aux", "wrist"]
    return list(cfg.dataset.img_keys)


def create_loaders(cfg, ddp: bool, batch_size: int):
    train_loader = create_dataset(
        cfg.dataset,
        horizon=cfg.horizon,
        ddp=ddp,
        batch_size=batch_size,
        n_workers=cfg.dataset.n_workers,
        split="train",
        return_video_frames=False,
        im_encoder_name=cfg.im_encoder.name,
        n_memory_frames=cfg.n_memory_frames,
        t_memory=cfg.t_memory,
        n_history=cfg.n_history,
    )
    valid_loader = create_dataset(
        cfg.dataset,
        horizon=max(2 * cfg.horizon, 24),
        ddp=ddp,
        batch_size=4,
        n_workers=1,
        max_trajectories=200,
        split="val",
        return_video_frames=True,
        im_encoder_name=cfg.im_encoder.name,
        n_memory_frames=cfg.n_memory_frames,
        t_memory=cfg.t_memory,
        n_history=cfg.n_history,
    )
    valid_video_loader = create_dataset(
        cfg.dataset,
        horizon=cfg.eval_video_frames,
        ddp=ddp,
        batch_size=4,
        n_workers=1,
        max_trajectories=None,
        split="val",
        return_video_frames=True,
        im_encoder_name=cfg.im_encoder.name,
        n_memory_frames=cfg.n_memory_frames,
        t_memory=cfg.t_memory,
        n_history=cfg.n_history,
    )
    return train_loader, valid_loader, valid_video_loader


def build_weaver(cfg):
    im_encoder, train_decoder = get_encoder(
        config=cfg.im_encoder,
        image_size=cfg.dataset.image_size,
        device="cuda",
    )
    task_encoder = get_task_encoder(config=None, device="cuda")

    image_size = cfg.dataset.image_size
    if isinstance(image_size, int):
        image_size = (image_size, image_size)

    model = WEAVER(
        img_keys=cfg.dataset.img_keys,
        im_encoder=im_encoder,
        train_decoder=train_decoder,
        task_encoder=task_encoder,
        n_history=cfg.n_history,
        n_horizon=cfg.horizon,
        config=cfg.model,
        use_precomputed_features=cfg.use_precomputed_features,
        n_states=cfg.dataset.n_states,
        n_actions=cfg.dataset.n_actions,
        image_size=image_size,
        device="cuda",
        n_memory_frames=cfg.n_memory_frames,
        t_memory=cfg.t_memory,
        inference_config=cfg.inference,
    ).to("cuda")
    model.ema.to("cuda")
    return model


def enable_activation_checkpointing(model, master_process: bool):
    if not master_process:
        return
    n_ckpt = sum(isinstance(module, STAttentionBlock) for module in model.modules())
    print(f"Activation checkpointing enabled on {n_ckpt} STAttentionBlock layers")


def save_validation_videos(raw_model, valid_video_iter, cfg, img_keys, vid_dir, step: int):
    raw_model.eval()
    n_vids = 4
    sample_idx = 0
    for _ in range(4):
        val_data = next(valid_video_iter)
        val_data = move_tensors_to_device(val_data, device="cuda")
        val_memory = {k: v[:n_vids] for k, v in val_data["memory"].items()} if "memory" in val_data else None

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, decoded_obs_pred = raw_model.generate_videos_full(
                obs={k: v[:n_vids] for k, v in val_data["obs"].items()},
                actions=val_data["actions"][:n_vids],
                instructions={k: v[:n_vids] for k, v in val_data["task"].items()},
                horizon=cfg.eval_horizon,
                memory=val_memory,
                bootstrap=cfg.eval_bootstrap,
            )

        t_pred = decoded_obs_pred[img_keys[0]].shape[1]
        for i in range(n_vids):
            gt_views = [
                rearrange(val_data["obs"][key][i:i + 1, :t_pred].float().cpu(), "b t c h w -> t h (b w) c")
                for key in img_keys
            ]
            pred_views = [
                rearrange(decoded_obs_pred[key][i:i + 1].float().cpu(), "b t c h w -> t h (b w) c")
                for key in img_keys
            ]
            video = np.concatenate([np.concatenate(gt_views, axis=2), np.concatenate(pred_views, axis=2)], axis=1)
            imageio.mimwrite(os.path.join(vid_dir, f"wm_valid_inference_step{step}_sample{sample_idx}.mp4"), (video * 255).astype(np.uint8), fps=5)
            sample_idx += 1

        del val_data, val_memory, decoded_obs_pred
        torch.cuda.empty_cache()
    raw_model.train()


def _shape_summary(name: str, value, indent: int = 0) -> list[str]:
    pad = " " * indent
    if isinstance(value, torch.Tensor):
        return [f"{pad}{name}: shape={tuple(value.shape)} dtype={value.dtype} device={value.device}"]
    if isinstance(value, dict):
        lines = [f"{pad}{name}:"]
        for key in sorted(value.keys()):
            lines.extend(_shape_summary(str(key), value[key], indent + 2))
        return lines
    if value is None:
        return [f"{pad}{name}: None"]
    return [f"{pad}{name}: {type(value).__name__}"]


def print_batch_shapes(data, step: int, accum_idx: int):
    print(f"=== FINETUNE BATCH SHAPES step={step} accum={accum_idx} ===", flush=True)
    for key in ["obs", "actions", "memory", "task", "rewards"]:
        for line in _shape_summary(key, data.get(key, None)):
            print(line, flush=True)
    print("=== END FINETUNE BATCH SHAPES ===", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    parser.add_argument("--mode", type=str, default="defaults", choices=["defaults", "debug"])
    parser.add_argument("--pretrained_dir", type=str, required=True)
    parser.add_argument("--finetune_suffix", type=str, default="finetune")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg, cfg_dict = parse_config(args)
    cfg_dict = inherit_pretrained_config(cfg_dict, load_pretrained_config(args.pretrained_dir))
    cfg_dict = apply_finetune_config(cfg_dict)
    cfg_dict = update_config(cfg_dict, dict(item.split("=", 1) for item in args.overrides))
    cfg = dict_to_namespace(cfg_dict)

    print("Arguments:", cfg)
    log_dir = os.path.join(cfg.scratch_dir, "logs")
    chkpt_save_dir = os.path.join(log_dir, f"chkpts_{args.finetune_suffix}")
    vid_dir = os.path.join(log_dir, "videos")
    os.makedirs(chkpt_save_dir, exist_ok=True)
    os.makedirs(vid_dir, exist_ok=True)

    ddp, _, local_rank, world_size, master_process = setup_ddp()
    img_keys = image_keys_from_cfg(cfg)
    train_loader, valid_loader, valid_video_loader = create_loaders(cfg, ddp, cfg.training.batch_size)

    model = build_weaver(cfg)
    finetune_ckpt = os.path.join(chkpt_save_dir, "checkpoint.pt")
    resume_finetune = os.path.exists(finetune_ckpt)
    load_dir = chkpt_save_dir if resume_finetune else args.pretrained_dir
    if not os.path.exists(os.path.join(load_dir, "checkpoint.pt")):
        raise FileNotFoundError(f"No checkpoint.pt found in {load_dir}")

    ckpt = load_checkpoint(load_dir, "cuda")
    model.load_state_dict(ckpt["model"])
    model.ema.load_state_dict(ckpt["ema"])
    if not resume_finetune:
        model.ema.ema.optimization_step = 0

    if cfg.use_compile:
        model = torch.compile(model)

    if cfg.use_activation_checkpointing:
        for module in model.modules():
            if isinstance(module, STAttentionBlock):
                module._use_checkpoint = True
        enable_activation_checkpointing(model, master_process)

    if ddp:
        model = DDP(model, device_ids=[local_rank])
    raw_model = model.module if ddp else model

    optimizer = raw_model.configure_optimizers(
        weight_decay=cfg.training.weight_decay,
        learning_rate=cfg.training.max_lr,
        betas=tuple(cfg.training.betas),
        device_type="cuda",
    )
    if resume_finetune:
        optimizer.load_state_dict(ckpt["optimizer"])
        step = ckpt["step"]
        print(f"Resuming finetune from step {step}")
    else:
        step = 0
        print(f"Starting finetune from pretrained checkpoint at {args.pretrained_dir}")

    def save_on_preempt(signum, frame):
        if master_process:
            save_checkpoint(chkpt_save_dir, model=getattr(raw_model, "_orig_mod", raw_model) if cfg.use_compile else raw_model, optimizer=optimizer, cfg=cfg, step=step)
        if ddp:
            dist.barrier()
        sys.exit(0)

    signal.signal(signal.SIGUSR1, save_on_preempt)

    if master_process:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters in WEAVER: {num_params}")
        if cfg.use_wandb:
            import wandb
            wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, sync_tensorboard=False, config=cfg_dict, name=f"{cfg.dataset.name}_{cfg.exp_name}_finetune", group=f"{cfg.dataset.name}/{cfg.exp_name}")

    train_iter = iter(cycle(train_loader))
    valid_video_iter = iter(cycle(valid_video_loader))
    data = next(train_iter)
    data = move_tensors_to_device(data, device="cuda")

    while step <= cfg.training.max_steps:
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated() / 1024**3
        optimizer.zero_grad()
        loss_accum = 0

        for accum_idx in range(cfg.training.gradient_accumulation_steps):
            data = move_tensors_to_device(next(train_iter), device="cuda")
            if master_process and step == 0 and accum_idx == 0:
                print_batch_shapes(data, step, accum_idx)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                total_loss, loss_log = model(
                    obs=data["obs"],
                    actions=data["actions"],
                    tasks=data["task"],
                    gt_rewards=data.get("rewards", None),
                    memory=data.get("memory", None),
                    update_rm=bool(step % cfg.model.rm_update_freq),
                )
            total_loss = total_loss / cfg.training.gradient_accumulation_steps
            loss_accum += total_loss.detach()
            if ddp:
                model.require_backward_grad_sync = accum_idx == cfg.training.gradient_accumulation_steps - 1
            total_loss.backward()

        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(step, cfg.training.warmup_steps, cfg.training.max_steps, cfg.training.max_lr, cfg.training.min_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        raw_model.ema.update()

        if step % cfg.log_freq == 0 and master_process:
            mem_after = torch.cuda.memory_allocated() / 1024**3
            mem_peak = torch.cuda.max_memory_allocated() / 1024**3
            mem_reserved = torch.cuda.memory_reserved() / 1024**3
            print(
                f"Step: {step} | Time: {time.time() - t0:.3f}s"
                f" | Total: {loss_accum.item():.6f}"
                f" | Flow: {loss_log['flow/Total Loss'].item():.6f}"
                f" | RM: {loss_log['RM loss'].item():.6f}"
                f" | V: {loss_log['Critic Loss'].item():.6f}"
                f" | Dec: {loss_log['decoder/Total Loss'].item():.6f}"
                f" | LR: {lr:.6f} | Norm: {norm:.6f}"
                f" | Mem(GB) alloc: {mem_after:.2f} peak: {mem_peak:.2f} reserved: {mem_reserved:.2f} before: {mem_before:.2f}"
            )
            if cfg.use_wandb:
                import wandb
                wandb.log({k: v.item() for k, v in loss_log.items()}, step=step)

        if step == 0 or step % cfg.valid_log_freq == 0:
            model.eval()
            raw_model.eval()
            if hasattr(valid_loader.sampler, "set_epoch"):
                valid_loader.sampler.set_epoch(step)
            valid_metrics = evaluate_with_metrics_ddp(
                model=model,
                raw_model=raw_model,
                val_dataloader=iter(valid_loader),
                img_keys=img_keys,
                device="cuda",
                master_process=master_process,
                world_size=world_size,
                horizon=cfg.eval_horizon,
                bootstrap=cfg.eval_bootstrap,
            )
            if master_process:
                metrics = {f"valid/{k}": v for k, v in valid_metrics.items()}
                print(f"Step: {step} | Valid Metrics: {metrics}")
                if cfg.use_wandb:
                    import wandb
                    wandb.log(metrics, step=step)
            raw_model.train()
            model.train()

        if cfg.save_model and step % cfg.ckpt_freq == 0 and master_process:
            save_checkpoint(chkpt_save_dir, model=getattr(raw_model, "_orig_mod", raw_model) if cfg.use_compile else raw_model, optimizer=optimizer, cfg=cfg, step=step)
            if step % 5000 == 0:
                save_checkpoint(chkpt_save_dir, model=getattr(raw_model, "_orig_mod", raw_model) if cfg.use_compile else raw_model, optimizer=optimizer, cfg=cfg, step=step, suffix=str(step))

        if step % cfg.video_log_freq == 0 and master_process:
            save_validation_videos(raw_model, valid_video_iter, cfg, img_keys, vid_dir, step)

        if ddp:
            dist.barrier()
        step += 1

    if cfg.save_model and master_process:
        save_checkpoint(chkpt_save_dir, model=getattr(raw_model, "_orig_mod", raw_model) if cfg.use_compile else raw_model, optimizer=optimizer, cfg=cfg, step=step)
    if ddp:
        destroy_process_group()
    print("Finetuning completed successfully.")


if __name__ == "__main__":
    main()
