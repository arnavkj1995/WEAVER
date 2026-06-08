"""Reflow post-training for WEAVER.

This trains a student WEAVER model to take one rectified-flow step from the same
noise sample ``x0`` to the frozen teacher rollout endpoint ``xhat1``. The teacher
is loaded with EMA weights and kept frozen; the student is initialized from the
same checkpoint unless resuming an existing reflow checkpoint.
"""

from __future__ import annotations

import argparse
import os
import re
import signal
import sys
import time
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path

import imageio
import numpy as np
import torch
import torch.distributed as dist
from einops import rearrange
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from .datasets import create_dataset
from .finetune import (
    image_keys_from_cfg,
    inherit_pretrained_config,
    load_pretrained_config,
)
from .utils.config import dict_to_namespace, parse_config, update_config
from .utils.eval import evaluate_with_metrics_ddp
from .utils.tools import cycle, get_lr, load_checkpoint, move_tensors_to_device, save_checkpoint
from .wm.encoders import get_encoder, get_task_encoder
from .wm.model import WEAVER
from .wm.nets import STAttentionBlock

torch.set_float32_matmul_precision("high")


def apply_posttrain_config(cfg_dict: dict) -> dict:
    cfg_dict = dict(cfg_dict)
    train_cfg = cfg_dict.get("posttrain_cfg", cfg_dict.get("finetune_cfg", {}))
    if train_cfg:
        cfg_dict.setdefault("training", {})
        cfg_dict["training"].update(train_cfg)
        cfg_name = "posttrain_cfg" if "posttrain_cfg" in cfg_dict else "finetune_cfg"
        print(f"Applied {cfg_name} to training: {train_cfg}")
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


def build_weaver_for_reflow(cfg, *, move_ema_to_cuda: bool = True) -> WEAVER:
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
    if move_ema_to_cuda:
        model.ema.to("cuda")
    return model


def freeze_model(model: torch.nn.Module) -> torch.nn.Module:
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


@contextmanager
def maybe_override_eval_generation(model: WEAVER, cfg):
    if not getattr(cfg, "eval_one_shot_chunk", False):
        yield
        return

    old_steps = model._inference_steps
    old_stagger = model._pyramid_stagger_width
    model._inference_steps = 1
    model._pyramid_stagger_width = 0
    model._pyramid_schedule_cache.clear()
    try:
        yield
    finally:
        model._inference_steps = old_steps
        model._pyramid_stagger_width = old_stagger
        model._pyramid_schedule_cache.clear()


def generate_teacher_rollout_with_noise(
    teacher_model: WEAVER,
    x1: dict[str, torch.Tensor],
    actions: torch.Tensor,
    memory=None,
    memory_tokens: torch.Tensor | None = None,
    inference_steps: int | None = None,
    stagger_width: int | None = None,
    schedule: str | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    if teacher_model._use_memory and memory_tokens is None:
        memory_tokens = teacher_model.encode_memory_obs(memory)

    x0 = teacher_model.sample_noise(x1)
    old_steps = teacher_model._inference_steps
    old_stagger = teacher_model._pyramid_stagger_width
    old_schedule = teacher_model._pyramid_schedule_type
    if inference_steps is not None:
        teacher_model._inference_steps = inference_steps
    if stagger_width is not None:
        teacher_model._pyramid_stagger_width = stagger_width
    if schedule is not None:
        teacher_model._pyramid_schedule_type = schedule
    teacher_model._pyramid_schedule_cache.clear()

    try:
        if teacher_model._diff_forcing:
            xhat1 = teacher_model._generate_latent_rollouts_autoregressive(
                x1, x0, actions, memory_tokens,
            )
        else:
            xhat1 = teacher_model._generate_latent_rollouts_lockstep(
                x1, x0, actions, memory_tokens,
            )
    finally:
        teacher_model._inference_steps = old_steps
        teacher_model._pyramid_stagger_width = old_stagger
        teacher_model._pyramid_schedule_type = old_schedule
        teacher_model._pyramid_schedule_cache.clear()

    return xhat1, x0


def _rollout_step_update(
    student_wm: WEAVER,
    xt: dict[str, torch.Tensor],
    x_pred: dict[str, torch.Tensor],
    step_t: torch.Tensor,
    dt: torch.Tensor,
    active: torch.Tensor,
) -> dict[str, torch.Tensor]:
    out = {}
    n_hist = student_wm._n_history
    for key in xt:
        if key not in x_pred:
            out[key] = xt[key]
            continue

        if key in student_wm._img_keys:
            active_view = active.view(1, -1, 1, 1).to(dtype=xt[key].dtype, device=xt[key].device)
            dt_view = dt.view(1, -1, 1, 1).to(dtype=xt[key].dtype, device=xt[key].device)
            t_view = step_t[:, n_hist:, None, None]
        else:
            active_view = active.view(1, -1, 1).to(dtype=xt[key].dtype, device=xt[key].device)
            dt_view = dt.view(1, -1, 1).to(dtype=xt[key].dtype, device=xt[key].device)
            t_view = step_t[:, n_hist:, None]

        future = xt[key][:, n_hist:]
        pred_future = x_pred[key][:, n_hist:]
        if student_wm._flow_loss.startswith("v-pred"):
            velocity = pred_future
        elif student_wm._flow_loss.startswith("x-pred"):
            velocity = (pred_future - future) / (1.0 - t_view).clamp(min=1e-2)
        else:
            raise NotImplementedError(f"Unsupported flow loss target: {student_wm._flow_loss}")

        next_future = future + active_view * velocity * dt_view
        out[key] = torch.cat([xt[key][:, :n_hist], next_future], dim=1)
    return out


def generate_student_rollout_from_noise(
    student_wm: WEAVER,
    x1_context: dict[str, torch.Tensor],
    x0: dict[str, torch.Tensor],
    actions: torch.Tensor,
    memory_tokens: torch.Tensor | None = None,
    *,
    inference_steps: int,
    stagger_width: int,
) -> dict[str, torch.Tensor]:
    B, T, _ = actions.size()
    n_hist = student_wm._n_history
    horizon = T - n_hist
    xt = {
        k: torch.cat([x1_context[k][:, :n_hist], x0[k][:, n_hist:]], dim=1)
        for k in x1_context
    }

    if student_wm._diff_forcing:
        old_steps = student_wm._inference_steps
        old_stagger = student_wm._pyramid_stagger_width
        student_wm._inference_steps = inference_steps
        student_wm._pyramid_stagger_width = stagger_width
        try:
            schedule = student_wm._build_pyramid_schedule(horizon).to(actions.device)
        finally:
            student_wm._inference_steps = old_steps
            student_wm._pyramid_stagger_width = old_stagger

        for m in range(schedule.shape[0] - 1):
            t_row = schedule[m]
            t_next_row = schedule[m + 1]
            active = t_row != t_next_row
            if not active.any():
                continue
            t_hist = torch.ones((B, n_hist), device=actions.device)
            step_t = torch.cat([t_hist, t_row.unsqueeze(0).expand(B, -1)], dim=1).float()
            x_pred = student_wm.wm(xt, actions, step_t, memory=memory_tokens)
            xt = _rollout_step_update(student_wm, xt, x_pred, step_t, t_next_row - t_row, active)
        return xt

    dt = 1.0 / inference_steps
    t_hist = torch.ones((B, n_hist), device=actions.device)
    step_t = torch.cat([t_hist, torch.zeros((B, horizon), device=actions.device)], dim=1).float()
    active = torch.ones(horizon, device=actions.device, dtype=torch.bool)
    for _ in range(inference_steps):
        x_pred = student_wm.wm(xt, actions, step_t, memory=memory_tokens)
        xt = _rollout_step_update(
            student_wm,
            xt,
            x_pred,
            step_t,
            torch.full((horizon,), dt, device=actions.device),
            active,
        )
        step_t[:, n_hist:] += dt
    return xt


class ReflowWrapper(torch.nn.Module):
    def __init__(self, student_wm: WEAVER, teacher_model: WEAVER):
        super().__init__()
        self.student_wm = student_wm
        self.teacher_model = teacher_model
        self.teacher_model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher_model.eval()
        return self

    def forward(self, obs, actions, tasks, gt_rewards, memory=None, update_rm: bool = True):
        student_wm = self.student_wm
        teacher_model = self.teacher_model
        losses = {}
        B, T, _ = actions.size()

        if gt_rewards is None:
            raise ValueError("ReflowWrapper.forward requires gt_rewards.")

        memory_tokens = None
        if student_wm._use_memory:
            memory_tokens = student_wm.encode_memory_obs(memory)
            if self.training and student_wm._history_noise_std > 0:
                memory_tokens = memory_tokens + torch.randn_like(memory_tokens) * student_wm._history_noise_std

        gt_loss_coeff = getattr(student_wm.config, "rectified_gt_loss_coeff", 0.0)
        rollout_loss_coeff = getattr(student_wm.config, "rectified_rollout_loss_coeff", 0.0)
        rollout_steps = int(getattr(student_wm.config, "rectified_student_rollout_steps", student_wm._inference_steps))
        rollout_stagger = int(getattr(student_wm.config, "rectified_student_rollout_stagger_width", student_wm._pyramid_stagger_width))
        teacher_steps = int(getattr(student_wm.config, "rectified_teacher_steps", teacher_model._inference_steps))
        teacher_stagger = int(getattr(student_wm.config, "rectified_teacher_stagger_width", teacher_model._pyramid_stagger_width))
        teacher_schedule = getattr(student_wm.config, "rectified_teacher_schedule", teacher_model._pyramid_schedule_type)

        with torch.no_grad():
            teacher_memory_tokens = teacher_model.encode_memory_obs(memory) if teacher_model._use_memory else None
            x1_data = teacher_model.encode_obs(obs)
            xhat1, x0 = generate_teacher_rollout_with_noise(
                teacher_model,
                x1_data,
                actions,
                memory=memory,
                memory_tokens=teacher_memory_tokens,
                inference_steps=teacher_steps,
                stagger_width=teacher_stagger,
                schedule=teacher_schedule,
            )
            xhat1 = {k: v.detach() for k, v in xhat1.items()}
            x0 = {k: v.detach() for k, v in x0.items()}
            x1_gt = {k: v.detach() for k, v in student_wm.encode_obs(obs).items()} if gt_loss_coeff > 0 else {}

        t = student_wm.sample_timestep(B, T)
        xt = student_wm.interpolate(xhat1, x0, t)

        def add_history_noise(x):
            if self.training and student_wm._history_noise_std > 0:
                for k in x:
                    hist = x[k][:, :student_wm._n_history]
                    future = x[k][:, student_wm._n_history:]
                    x[k] = torch.cat([hist + torch.randn_like(hist) * student_wm._history_noise_std, future], dim=1)
            return x

        xt = add_history_noise(xt)
        x_pred = student_wm.wm(xt, actions, t, memory=memory_tokens)

        for key in x_pred:
            if key in student_wm._img_keys:
                t_view = rearrange(t, "b t -> b t 1 1")
            else:
                t_view = rearrange(t, "b t -> b t 1")
            loss = student_wm.compute_flow_loss(xhat1[key], x0[key], x_pred[key], t_view)
            loss = loss[:, student_wm._n_history:].mean(dim=-1)
            losses[f"flow/{key}"] = loss.mean()

        if gt_loss_coeff > 0:
            xt_gt = add_history_noise(student_wm.interpolate(x1_gt, x0, t))
            x_pred_gt = student_wm.wm(xt_gt, actions, t, memory=memory_tokens)
            for key in x_pred_gt:
                if key in student_wm._img_keys:
                    t_view = rearrange(t, "b t -> b t 1 1")
                else:
                    t_view = rearrange(t, "b t -> b t 1")
                gt_loss = student_wm.compute_flow_loss(x1_gt[key], x0[key], x_pred_gt[key], t_view)
                gt_loss = gt_loss[:, student_wm._n_history:].mean(dim=-1)
                losses[f"flow_gt/{key}"] = gt_loss.mean()

        device = actions.device
        flow_loss = sum(losses[f"flow/{k}"] for k in student_wm._img_keys)
        if "flow/states" in losses:
            flow_loss = flow_loss + student_wm._state_loss_scale * losses["flow/states"]

        gt_flow_loss = torch.zeros((), device=device)
        if gt_loss_coeff > 0:
            gt_flow_loss = sum(losses[f"flow_gt/{k}"] for k in student_wm._img_keys)
            if "flow_gt/states" in losses:
                gt_flow_loss = gt_flow_loss + student_wm._state_loss_scale * losses["flow_gt/states"]

        rollout_loss = torch.zeros((), device=device)
        student_rollout = None
        if rollout_loss_coeff > 0:
            student_rollout = generate_student_rollout_from_noise(
                student_wm,
                xhat1,
                x0,
                actions,
                memory_tokens=memory_tokens,
                inference_steps=rollout_steps,
                stagger_width=rollout_stagger,
            )
            for key in student_rollout:
                if key not in xhat1:
                    continue
                rollout_key_loss = torch.nn.functional.mse_loss(
                    student_rollout[key],
                    xhat1[key],
                    reduction="none",
                )
                rollout_key_loss = rollout_key_loss[:, student_wm._n_history:].mean(dim=-1).mean()
                losses[f"rollout/{key}"] = rollout_key_loss
                rollout_loss = rollout_loss + (student_wm._state_loss_scale if key == "states" else 1.0) * rollout_key_loss

        task_embed = student_wm.encode_task(tasks).detach()
        endpoint_source = student_rollout
        xhat1_pred = {}
        for key in xhat1:
            if endpoint_source is not None and key in endpoint_source:
                xhat1_pred[key] = endpoint_source[key].detach()
            elif key in x_pred:
                if student_wm._flow_loss.startswith("v-pred"):
                    xhat1_pred[key] = (x0[key] + x_pred[key]).detach()
                elif student_wm._flow_loss.startswith("x-pred"):
                    xhat1_pred[key] = x_pred[key].detach()
                else:
                    raise NotImplementedError(f"Unsupported flow loss target: {student_wm._flow_loss}")

        xhat1_detached = {k: v.detach() for k, v in xhat1.items() if k in xhat1_pred}
        obs_both = {k: torch.cat([xhat1_detached[k], xhat1_pred[k]], dim=0) for k in xhat1_pred}
        actions_both = torch.cat([actions, actions], dim=0)
        tasks_both = torch.cat([task_embed, task_embed], dim=0)
        rewards_both = torch.cat([gt_rewards, gt_rewards], dim=0)

        pred_rewards, rm_loss = student_wm.rm.compute_loss(
            obs=obs_both,
            actions=actions_both,
            tasks=tasks_both,
            gt_rewards=rewards_both,
        )
        critic_loss = student_wm.critic.compute_loss(
            obs=obs_both,
            actions=actions_both,
            tasks=tasks_both,
            rewards=rewards_both,
        )

        total_loss = (
            flow_loss
            + gt_loss_coeff * gt_flow_loss
            + rollout_loss_coeff * rollout_loss
            + 0.0 * pred_rewards.mean()
            + 0.0 * rm_loss
            + 0.0 * critic_loss
        )

        log = {k: v.detach() for k, v in losses.items()}
        log.update({
            "flow/Total Loss": flow_loss.detach(),
            "flow_gt/Total Loss": gt_flow_loss.detach(),
            "flow_gt/Coeff": torch.tensor(gt_loss_coeff, device=device),
            "rollout/Total Loss": rollout_loss.detach(),
            "rollout/Coeff": torch.tensor(rollout_loss_coeff, device=device),
            "rollout/Steps": torch.tensor(rollout_steps, device=device),
            "rollout/Stagger Width": torch.tensor(rollout_stagger, device=device),
            "teacher/Steps": torch.tensor(teacher_steps, device=device),
            "teacher/Stagger Width": torch.tensor(teacher_stagger, device=device),
            "teacher/Schedule Is Cosine": torch.tensor(float(teacher_schedule == "cosine"), device=device),
            "RM loss": rm_loss.detach(),
            "Critic Loss": critic_loss.detach(),
            "decoder/Total Loss": torch.zeros((), device=device),
            "Temporal/Total Loss": torch.zeros((), device=device),
            "Total Loss": total_loss.detach(),
        })
        return total_loss, log


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
        max_trajectories=16,
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


def save_validation_videos(raw_model: WEAVER, valid_video_iter, cfg, img_keys, vid_dir: str, step: int):
    raw_model.eval()
    eval_bootstrap = cfg.eval_horizon if getattr(cfg, "eval_one_shot_chunk", False) else getattr(cfg, "eval_bootstrap", None)
    n_vids = 4
    sample_idx = 0
    with maybe_override_eval_generation(raw_model, cfg):
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
                    bootstrap=eval_bootstrap,
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
                imageio.mimwrite(
                    os.path.join(vid_dir, f"wm_valid_inference_step{step}_sample{sample_idx}.mp4"),
                    (video * 255).astype(np.uint8),
                    fps=5,
                )
                sample_idx += 1

            del val_data, val_memory, decoded_obs_pred
            torch.cuda.empty_cache()
    raw_model.train()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    parser.add_argument("--mode", type=str, default="defaults", choices=["defaults", "debug"])
    parser.add_argument("--pretrained_dir", type=str, required=True)
    parser.add_argument("--pretrained_ckpt_name", type=str, default="checkpoint.pt")
    parser.add_argument("--finetune_suffix", type=str, default="reflow")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg, cfg_dict = parse_config(args)
    cfg_dict = inherit_pretrained_config(cfg_dict, load_pretrained_config(args.pretrained_dir))
    cfg_dict = apply_posttrain_config(cfg_dict)
    cfg_dict = update_config(cfg_dict, dict(item.split("=", 1) for item in args.overrides))
    cfg = dict_to_namespace(cfg_dict)

    print("Arguments:", cfg)
    log_dir = os.path.join(cfg.scratch_dir, "logs")
    chkpt_save_dir = os.path.join(log_dir, f"chkpts_{args.finetune_suffix}")
    vid_dir = os.path.join(log_dir, "videos")
    os.makedirs(chkpt_save_dir, exist_ok=True)
    os.makedirs(vid_dir, exist_ok=True)

    finetune_ckpt = os.path.join(chkpt_save_dir, "checkpoint.pt")
    resume_finetune = os.path.exists(finetune_ckpt)
    load_dir = chkpt_save_dir if resume_finetune else args.pretrained_dir
    load_ckpt_name = "checkpoint.pt" if resume_finetune else args.pretrained_ckpt_name
    if not os.path.exists(os.path.join(load_dir, load_ckpt_name)):
        raise FileNotFoundError(f"No checkpoint found at {load_dir}/{load_ckpt_name}")

    ddp, _, local_rank, world_size, master_process = setup_ddp()
    img_keys = image_keys_from_cfg(cfg)
    train_loader, valid_loader, valid_video_loader = create_loaders(cfg, ddp, cfg.training.batch_size)

    student_wm = build_weaver_for_reflow(cfg)
    print(f"Loading student checkpoint from {load_dir}/{load_ckpt_name}")
    ckpt = load_checkpoint(load_dir, "cuda", checkpoint_name=load_ckpt_name)
    student_wm.load_state_dict(ckpt["model"])
    student_wm.ema.load_state_dict(ckpt["ema"])
    student_wm.ema.to("cuda")

    print(f"Loading frozen teacher checkpoint from {args.pretrained_dir}/{args.pretrained_ckpt_name}")
    teacher_ckpt = load_checkpoint(args.pretrained_dir, "cpu", checkpoint_name=args.pretrained_ckpt_name)
    teacher_wm = build_weaver_for_reflow(cfg, move_ema_to_cuda=False)
    teacher_wm.load_state_dict(teacher_ckpt["model"])
    teacher_wm.ema.load_state_dict(teacher_ckpt["ema"])
    teacher_wm.ema.apply_to(teacher_wm)
    teacher_wm.ema = None
    teacher_wm = freeze_model(teacher_wm)

    if not resume_finetune:
        student_wm.load_state_dict(teacher_wm.state_dict())
        student_wm.ema.load_state_dict(teacher_ckpt["ema"])
        student_wm.ema.to("cuda")
        student_wm.ema.ema.optimization_step = 0

    model = ReflowWrapper(student_wm, teacher_wm)
    if master_process:
        print("Using reflow post-training objective: student x0 -> frozen teacher xhat1")
        if cfg.use_compile:
            print("Not compiling ReflowWrapper; the inner FlowWM.forward compile boundary is still active.")

    if cfg.use_activation_checkpointing:
        for module in student_wm.modules():
            if isinstance(module, STAttentionBlock):
                module._use_checkpoint = True
        if master_process:
            n_ckpt = sum(isinstance(module, STAttentionBlock) for module in student_wm.modules())
            print(f"Activation checkpointing enabled on {n_ckpt} STAttentionBlock layers")

    if ddp:
        model = DDP(model, device_ids=[local_rank])
    raw_wrapper = model.module if ddp else model
    raw_wm = raw_wrapper.student_wm

    optimizer = raw_wm.configure_optimizers(
        weight_decay=cfg.training.weight_decay,
        learning_rate=cfg.training.max_lr,
        betas=tuple(cfg.training.betas),
        device_type="cuda",
    )
    if resume_finetune:
        optimizer.load_state_dict(ckpt["optimizer"])
        step = ckpt["step"]
        print(f"Resuming reflow from step {step}")
    else:
        step = 0
        print(f"Starting reflow from teacher checkpoint at {args.pretrained_dir}")

    def save_on_preempt(signum, frame):
        if master_process:
            print(f"Caught signal {signum} at step {step}; saving checkpoint.")
            save_checkpoint(chkpt_save_dir, model=raw_wm, optimizer=optimizer, cfg=cfg, step=step, save_config=False, atomic=False)
        if ddp:
            dist.barrier()
        sys.exit(0)

    signal.signal(signal.SIGUSR1, save_on_preempt)

    if master_process:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters in reflow wrapper: {num_params}")
        if cfg.use_wandb:
            import wandb
            wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                sync_tensorboard=False,
                config=cfg_dict,
                name=f"{cfg.dataset.name}_{cfg.exp_name}_reflow",
                group=f"{cfg.dataset.name}/{cfg.exp_name}",
            )

    train_iter = iter(cycle(train_loader))
    valid_video_iter = iter(cycle(valid_video_loader))
    eval_bootstrap = cfg.eval_horizon if getattr(cfg, "eval_one_shot_chunk", False) else getattr(cfg, "eval_bootstrap", None)
    milestone_cfg = getattr(cfg, "checkpoint_milestones", "")
    if isinstance(milestone_cfg, (list, tuple, set)):
        checkpoint_milestones = {int(x) for x in milestone_cfg}
    else:
        checkpoint_milestones = {int(x) for x in re.findall(r"\d+", str(milestone_cfg))}

    while step <= cfg.training.max_steps:
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated() / 1024**3
        optimizer.zero_grad()
        loss_accum = 0

        for accum_idx in range(cfg.training.gradient_accumulation_steps):
            data = move_tensors_to_device(next(train_iter), device="cuda")
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                total_loss, loss_log = model(
                    obs=data["obs"],
                    actions=data["actions"],
                    tasks=data["task"],
                    gt_rewards=data["rewards"],
                    memory=data.get("memory", None),
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
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.step()
        raw_wm.ema.update()

        step_time = time.time() - t0
        mem_after = torch.cuda.memory_allocated() / 1024**3
        mem_peak = torch.cuda.max_memory_allocated() / 1024**3
        mem_reserved = torch.cuda.memory_reserved() / 1024**3

        if step % cfg.log_freq == 0 and master_process:
            log_str = (
                f"Step: {step} | Time: {step_time:.3f}s"
                f" | Total: {loss_accum.item():.6f}"
                f" | Flow: {loss_log['flow/Total Loss'].item():.6f}"
                f" | Rollout: {loss_log['rollout/Total Loss'].item():.6f}"
                f" | RM: {loss_log['RM loss'].item():.6f}"
                f" | V: {loss_log['Critic Loss'].item():.6f}"
                f" | LR: {lr:.6f} | Norm: {norm:.6f}"
                f" | Mem(GB) alloc: {mem_after:.2f} peak: {mem_peak:.2f}"
                f" reserved: {mem_reserved:.2f} before: {mem_before:.2f}"
            )
            print(log_str)
            if cfg.use_wandb:
                import wandb
                wandb.log({k: v.item() for k, v in loss_log.items()}, step=step)

        if cfg.valid_log_freq > 0 and (step == 0 or step % cfg.valid_log_freq == 0):
            model.eval()
            raw_wm.eval()
            if hasattr(valid_loader.sampler, "set_epoch"):
                valid_loader.sampler.set_epoch(step)
            valid_iter = iter(valid_loader)
            with maybe_override_eval_generation(raw_wm, cfg):
                valid_vid_metrics = evaluate_with_metrics_ddp(
                    model=model,
                    raw_model=raw_wm,
                    val_dataloader=valid_iter,
                    img_keys=img_keys,
                    device="cuda",
                    master_process=master_process,
                    world_size=world_size if ddp else 1,
                    horizon=cfg.eval_horizon,
                    bootstrap=eval_bootstrap,
                )
            if master_process:
                valid_metrics = {f"valid/{k}": v for k, v in valid_vid_metrics.items()}
                print(f"Step: {step} | Valid Metrics: {valid_metrics}")
                if cfg.use_wandb:
                    import wandb
                    wandb.log(valid_metrics, step=step)
            raw_wm.train()
            model.train()

        if cfg.save_model and cfg.ckpt_freq > 0 and step > 0 and step % cfg.ckpt_freq == 0 and master_process:
            save_checkpoint(chkpt_save_dir, model=raw_wm, optimizer=optimizer, cfg=cfg, step=step, save_config=False, atomic=False)

        if cfg.save_model and step in checkpoint_milestones and master_process:
            save_checkpoint(chkpt_save_dir, model=raw_wm, optimizer=optimizer, cfg=cfg, step=step, suffix=f"_step{step}")

        if cfg.video_log_freq > 0 and step % cfg.video_log_freq == 0 and master_process:
            save_validation_videos(raw_wm, valid_video_iter, cfg, img_keys, vid_dir, step)

        if ddp:
            dist.barrier()
        step += 1

    if cfg.save_model and master_process:
        save_checkpoint(chkpt_save_dir, model=raw_wm, optimizer=optimizer, cfg=cfg, step=step, save_config=False, atomic=False)

    if ddp:
        destroy_process_group()
    print("Reflow post-training completed successfully.")


if __name__ == "__main__":
    main()
