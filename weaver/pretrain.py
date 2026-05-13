import argparse
import os
import time
from datetime import timedelta

import imageio
import numpy as np
import torch
import torch.distributed as dist
from einops import rearrange
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from .utils.config import parse_config
from .wm.encoders import get_encoder, get_task_encoder
from .wm.model import WEAVER
from .wm.nets import STAttentionBlock
from .datasets import create_dataset
from .utils.tools import load_checkpoint, save_checkpoint, get_lr, cycle, move_tensors_to_device
from .utils.eval import evaluate_with_metrics_ddp

torch.set_float32_matmul_precision('high')


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    parser.add_argument("--mode", type=str, default="defaults", choices=["defaults", "debug"])
    parser.add_argument("overrides", nargs="*", help="Override config params, e.g. training.lr=1e-4 model.num_layers=8")
    args = parser.parse_args()

    # Load and update
    cfg, cfg_dict = parse_config(args)

    print ('Arguments:', cfg)
    log_dir = os.path.join(cfg.scratch_dir, 'logs')
    chkpt_dir = os.path.join(log_dir, 'chkpts')
    vid_dir = os.path.join(log_dir, 'videos')
    
    os.makedirs(chkpt_dir, exist_ok=True)
    os.makedirs(vid_dir, exist_ok=True)
    
    # various inits, derived attributes, I/O setup
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    if ddp:
        init_process_group(backend='nccl', timeout=timedelta(hours=2))
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank  # each process gets a different seed
        # world_size number of processes will be training simultaneously, so we can scale
        # down the desired gradient accumulation iterations per process proportionally
        # assert gradient_accumulation_steps % ddp_world_size == 0
        # gradient_accumulation_steps //= ddp_world_size
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 0
        ddp_rank = 0
        ddp_world_size = 1

    B = cfg.training.batch_size
    if cfg.dataset.sample_aux_from_left_right:
        IMG_KEYS = ['aux', 'wrist']  # aux = random left/right per trajectory
    else:
        IMG_KEYS = cfg.dataset.img_keys


    # exp_dataloader = create_dataset(
    #     cfg.dataset,
    #     horizon=cfg.horizon + cfg.n_history,
    #     ddp=ddp,
    #     batch_size=B // 2,
    #     n_workers=cfg.dataset.n_workers,
    #     #max_trajectories=224,
    #     split='exp'
    # )
    policy_dataloader = create_dataset(
        cfg.dataset,
        horizon=cfg.horizon,
        ddp=ddp,
        batch_size=B,  # // 2,
        n_workers=cfg.dataset.n_workers,
        #max_trajectories=1000,
        split='train',
        return_video_frames=False,
        im_encoder_name=cfg.im_encoder.name,
        n_memory_frames=cfg.n_memory_frames,
        t_memory=cfg.t_memory,
        n_history=cfg.n_history,
    )
    # train_dataloader_short = create_dataset(
    #     cfg.dataset,
    #     horizon=cfg.horizon + cfg.n_history,
    #     ddp=ddp,
    #     batch_size=4,
    #     max_trajectories=100,
    #     split='train'
    # )

    # val_dataloader = train_dataloader
    # valid_dataloader = create_dataset(
    #     cfg.dataset,
    #     horizon=cfg.horizon + cfg.n_history,
    #     ddp=ddp,
    #     batch_size=2,
    #     n_workers=1,
    #     max_trajectories=100,
    #     split='val'
    # )
    valid_video_dataloader = create_dataset(
        cfg.dataset,
        horizon=cfg.eval_video_frames,
        ddp=ddp,
        batch_size=4,
        n_workers=1,
        max_trajectories=None,
        split='val',
        return_video_frames=True,
        im_encoder_name=cfg.im_encoder.name,
        n_memory_frames=cfg.n_memory_frames,
        t_memory=cfg.t_memory,
        n_history=cfg.n_history,
    )
    valid_dataloader = create_dataset(
        cfg.dataset,
        horizon=max(2 * cfg.horizon,16),
        ddp=ddp,
        batch_size=4,
        n_workers=1,
        max_trajectories=200, #None,
        split='val',
        return_video_frames=True,
        im_encoder_name=cfg.im_encoder.name,
        n_memory_frames=cfg.n_memory_frames,
        t_memory=cfg.t_memory,
        n_history=cfg.n_history,
    )

    # Create image encoder
    
    im_encoder, train_decoder = get_encoder(
        config=cfg.im_encoder,
        image_size=cfg.dataset.image_size,
        device='cuda',
    )

    # Log spatial configuration (automatically derived from encoder)
    if hasattr(im_encoder, 'spatial_size'):
        spatial_size = im_encoder.spatial_size
        image_size = cfg.dataset.image_size
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        n_patches = (image_size[0] // 8 // spatial_size) * (image_size[1] // 8 // spatial_size)
        print(f"Spatial config: image_size={image_size}, spatial_size={spatial_size} → {n_patches} patches/image")

    # T5 model for instruction encoding
    task_encoder = get_task_encoder(
        config=None,  # cfg.task_encoder,
        device='cuda'
    )

    # Get image_size from config (may be int or tuple)
    image_size = cfg.dataset.image_size
    if isinstance(image_size, int):
        image_size = (image_size, image_size)

    # Define the world model
    wm = WEAVER(
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
        device='cuda',
        n_memory_frames=cfg.n_memory_frames,
        t_memory=cfg.t_memory,
        inference_config=cfg.inference,
    ).to('cuda')
    wm.ema.to('cuda')

    if cfg.resume and not os.path.exists(os.path.join(chkpt_dir, 'checkpoint.pt')):
        print ("No checkpoint found. Starting from scratch...")
        cfg.resume = False

    if cfg.resume:
        ckpt = load_checkpoint(chkpt_dir, 'cuda')
        wm.load_state_dict(ckpt["model"])
        wm.ema.load_state_dict(ckpt['ema'])

    if cfg.use_compile:
        # wm.wm = torch.compile(wm.wm)
        wm = torch.compile(wm)

    # Apply activation checkpointing to reduce memory usage
    use_act_ckpt = cfg.use_activation_checkpointing
    if use_act_ckpt:
        n_ckpt = 0
        for module in wm.modules():
            if isinstance(module, STAttentionBlock):
                module._use_checkpoint = True
                n_ckpt += 1
        if master_process:
            print(f"Activation checkpointing enabled on {n_ckpt} STAttentionBlock layers")

    if ddp:
        wm = DDP(wm, device_ids=[ddp_local_rank])
    raw_wm = wm.module if ddp else wm

    # Extract training parameters
    gradient_accum_steps = cfg.training.gradient_accumulation_steps
    max_steps = cfg.training.max_steps
    warmup_steps = cfg.training.warmup_steps
    max_lr = cfg.training.max_lr
    min_lr = cfg.training.min_lr

    # Optimizer hyperparameters from config
    weight_decay = cfg.training.weight_decay
    betas = tuple(cfg.training.betas)

    optimizer = raw_wm.configure_optimizers(
        weight_decay=weight_decay,
        learning_rate=max_lr,
        betas=betas,
        device_type='cuda'
    )
    
    if cfg.resume:
        optimizer.load_state_dict(ckpt['optimizer'])
        step = ckpt['step']
    else:
        step = 1

    if master_process:
        # Print number of trainable parameters in wm
        num_params = sum(p.numel() for p in wm.parameters() if p.requires_grad)
        print(f'Number of trainable parameters in WM: {num_params}')
        
        if cfg.use_wandb:
            import wandb

            run_wandb = wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                sync_tensorboard=False,
                config=cfg_dict,
                name=f"{cfg.dataset.name}_{cfg.exp_name}",
                group=f"{cfg.dataset.name}/{cfg.exp_name}",
            )

    # exp_dataloader_iter = iter(cycle(exp_dataloader))
    policy_dataloader_iter = iter(cycle(policy_dataloader))
    valid_video_dataloader_iter = iter(cycle(valid_video_dataloader))
    
    data = next(policy_dataloader_iter)
    data = move_tensors_to_device(data, device='cuda')
    # step = 0
    # data = sample_and_merge_batch(
    #     exp_dataloader_iter,
    #     policy_dataloader_iter
    # )

    while step <= max_steps:
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated() / 1024**3  # GB
        optimizer.zero_grad()

        loss_accum = 0
        for _accum in range(gradient_accum_steps):
            
            # print (" Starting nthe loop again")
            data = next(policy_dataloader_iter)
            data = move_tensors_to_device(data, device='cuda')
            
            # print (" Loop running ")
            # data = sample_and_merge_batch(
            #     exp_dataloader_iter,
            #     policy_dataloader_iter
            # )
            
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                total_loss, loss_log = wm(
                    obs=data['obs'],
                    actions=data['actions'],
                    tasks=data['task'],
                    gt_rewards=data.get('rewards', None),
                    memory=data.get('memory', None),
                    update_rm=bool(step % cfg.model.rm_update_freq),
                )

            total_loss = total_loss / gradient_accum_steps
            loss_accum += total_loss.detach()

            if ddp:
                wm.require_backward_grad_sync = (_accum == gradient_accum_steps - 1)
            total_loss.backward()

        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        norm = torch.nn.utils.clip_grad_norm_(wm.parameters(), 1.0)

        lr = get_lr(step, warmup_steps, max_steps, max_lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.step()
        raw_wm.ema.update()

        # Memory stats
        step_time = time.time() - t0
        mem_after = torch.cuda.memory_allocated() / 1024**3  # GB
        mem_peak = torch.cuda.max_memory_allocated() / 1024**3  # GB
        mem_reserved = torch.cuda.memory_reserved() / 1024**3  # GB

        if step % cfg.log_freq == 0 and master_process:
            log_str = (
                f'Step: {step} | Time: {step_time:.3f}s'
                f' | Total: {loss_accum.item():.6f}'
                f' | Flow: {loss_log["flow/Total Loss"].item():.6f}'
                f' | RM: {loss_log["RM loss"].item():.6f}'
                f' | V: {loss_log["Critic Loss"].item():.6f}'
                f' | Dec: {loss_log["decoder/Total Loss"].item():.6f}'
            )
            log_str += f' | LR: {lr:.6f} | Norm: {norm:.6f}'
            log_str += (
                f' | Mem(GB) alloc: {mem_after:.2f}'
                f' peak: {mem_peak:.2f}'
                f' reserved: {mem_reserved:.2f}'
                f' before: {mem_before:.2f}'
            )
            print(log_str)

            if cfg.use_wandb:
                wandb_log = {k: v.item() for k, v in loss_log.items()}
                wandb.log(wandb_log, step=step)


        if step % cfg.valid_log_freq == 0:
            wm.eval()
            raw_wm.eval()
            
            t0 = time.time()
                        
            print ('Evaluating on validation set...')
            if hasattr(valid_dataloader.sampler, 'set_epoch'):
                valid_dataloader.sampler.set_epoch(step)
            valid_dataloader_iter = iter(valid_dataloader)

            valid_vid_metrics = evaluate_with_metrics_ddp(
                model=wm,
                raw_model=raw_wm,
                val_dataloader=valid_dataloader_iter,
                img_keys=IMG_KEYS,
                device='cuda',
                master_process=master_process,
                world_size=ddp_world_size if ddp else 1,
                horizon=cfg.eval_horizon,
                bootstrap=cfg.eval_bootstrap,
            )
            
            valid_metrics = {f'valid/{k}': v for k, v in valid_vid_metrics.items()}

            if master_process:
                print(f'Step: {step} | Time taken: {time.time() - t0:.3f}s | Valid Metrics: {valid_metrics}')
                if cfg.use_wandb:
                    wandb.log(valid_metrics, step=step)
                
            raw_wm.train()
            wm.train()
        
        if cfg.save_model and step % cfg.ckpt_freq == 0 and master_process:
            save_checkpoint(
                chkpt_dir,
                model=getattr(raw_wm, "_orig_mod", raw_wm) if cfg.use_compile else raw_wm,
                optimizer=optimizer,
                cfg=cfg,
                step=step
            )
            
        if step % cfg.video_log_freq == 0 and master_process:
            raw_wm.eval()  # Disable SPRINT token dropping for inference
            n_vids=4 # B is batch - size

            with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                train_memory = {k: v[:n_vids] for k, v in data['memory'].items()} if 'memory' in data else None
                decoded_obs, decoded_obs_pred, _ = raw_wm.generate_videos(
                    x1={k: v[:n_vids] for k, v in data['obs'].items()},
                    actions=data['actions'][:n_vids],
                    instructions={k: v[:n_vids] for k, v in data['task'].items()},
                    encode=True,
                    memory=train_memory,
                )

            for i in range(n_vids):
                recon_views = [
                    rearrange(decoded_obs[key][i:i+1].float().detach().cpu(), 'b t c h w -> t h (b w) c')
                    for key in IMG_KEYS
                ]
                pred_views = [
                    rearrange(decoded_obs_pred[key][i:i+1].float().detach().cpu(), 'b t c h w -> t h (b w) c')
                    for key in IMG_KEYS
                ]
                recon_vid = np.concatenate(recon_views, axis=2)  # concat along width
                pred_vid = np.concatenate(pred_views, axis=2)   # concat along width

                video = np.concatenate([recon_vid, pred_vid], axis=1)  # GT top, pred bottom
                video = (video * 255).astype(np.uint8)
                
                imageio.mimwrite(os.path.join(vid_dir, f'wm_train_inference_step{step}_sample{i}.mp4'), video, fps=5)
                # imageio.mimwrite(f'vid_pretrain/wm_train_inference_step{step}_sample{i}.mp4', video, fps=5)
            
            print(" Video saved ")

            del decoded_obs, decoded_obs_pred
            torch.cuda.empty_cache()

            # continue
            sample_idx = 0
            for _ in range(4):  # 4 batches × n_vids=4 = 16 videos total
                val_data = next(valid_video_dataloader_iter)
                val_data = move_tensors_to_device(val_data, device='cuda')

                start_time = time.time()
                n_vids=4
                val_memory = {k: v[:n_vids] for k, v in val_data['memory'].items()} if 'memory' in val_data else None
                
                with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    _, decoded_obs_pred = raw_wm.generate_videos_full(
                        obs={k: v[:n_vids] for k, v in val_data['obs'].items()},
                        actions=val_data['actions'][:n_vids],
                        instructions={k: v[:n_vids] for k, v in val_data['task'].items()},
                        horizon=cfg.eval_horizon,
                        memory=val_memory,
                        bootstrap=cfg.eval_bootstrap,
                    )
                
                print (f"Time taken for video gen: {time.time() - start_time:.3f}s")

                T_pred = decoded_obs_pred[IMG_KEYS[0]].shape[1]
                for i in range(n_vids):
                    # Concatenate all camera views side by side for this sample
                    orig_views = [
                        rearrange(val_data['obs'][key][i:i+1, :T_pred].float().detach().cpu(), 'b t c h w -> t h (b w) c')
                        for key in IMG_KEYS
                    ]
                    pred_views = [
                        rearrange(decoded_obs_pred[key][i:i+1].float().detach().cpu(), 'b t c h w -> t h (b w) c')
                        for key in IMG_KEYS
                    ]
                    orig_vid = np.concatenate(orig_views, axis=2)  # concat along width
                    pred_vid = np.concatenate(pred_views, axis=2)  # concat along width

                    video = np.concatenate([orig_vid, pred_vid], axis=1)  # concat along height
                    video = (video * 255).astype(np.uint8)
                    imageio.mimwrite(os.path.join(vid_dir, f'wm_valid_inference_step{step}_sample{sample_idx}.mp4'), video, fps=5)
                    # imageio.mimwrite(f'vid_pretrain/wm_valid_inference_step{step}_sample{sample_idx}.mp4', video, fps=5)
                    sample_idx += 1
            
                print(" Video saved ")
                    
                del val_data, val_memory, decoded_obs_pred
                torch.cuda.empty_cache()

            raw_wm.train()  # Re-enable training mode (SPRINT token dropping)

        # Sync all ranks after checkpoint saving and video generation (rank 0 only)
        # to prevent NCCL timeouts at the next training step's ALLREDUCE
        if ddp:
            dist.barrier()

        step += 1
    
    if cfg.save_model and master_process:
        save_checkpoint(
            chkpt_dir,
            model=getattr(raw_wm, "_orig_mod", raw_wm) if cfg.use_compile else raw_wm,
            optimizer=optimizer,
            cfg=cfg,
            step=step
        )
        
    if ddp:
        destroy_process_group()
    # if accelerator.is_main_process:
    print("Code executed successfully.")


if __name__ == "__main__":
    main()
