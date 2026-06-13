# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Main training loop."""

import os
import subprocess
import time
import copy
import json
import pickle
import psutil
import numpy as np
import torch
import dnnlib
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import misc
from fid import calculate_inception_stats, calculate_fid_from_inception_stats

#----------------------------------------------------------------------------

def update_target_network_hard(model, target_model):
    """Update target network using hard copy"""
    for param, target_param in zip(model.parameters(), target_model.parameters()):
        target_param.data.copy_(param.data)

def update_target_network_ema(model, target_model, tau=0.999):
    """Update target network using EMA (Exponential Moving Average)"""
    for param, target_param in zip(model.parameters(), target_model.parameters()):
        target_param.data.copy_(tau * target_param.data + (1.0 - tau) * param.data)

def training_loop(
    run_dir             = '.',      # Output directory.
    dataset_kwargs      = {},       # Options for training set.
    data_loader_kwargs  = {},       # Options for torch.utils.data.DataLoader.
    network_kwargs      = {},       # Options for model and preconditioning.
    loss_kwargs         = {},       # Options for loss function.
    optimizer_kwargs    = {},       # Options for optimizer.
    augment_kwargs      = None,     # Options for augmentation pipeline, None = disable.
    seed                = 0,        # Global random seed.
    batch_size          = 512,      # Total batch size for one training iteration.
    batch_gpu           = None,     # Limit batch size per GPU, None = no limit.
    total_kimg          = 200000,   # Training duration, measured in thousands of training images.
    ema_halflife_kimg   = 500,      # Half-life of the exponential moving average (EMA) of model weights.
    ema_rampup_ratio    = 0.05,     # EMA ramp-up coefficient, None = no rampup.
    lr_rampup_kimg      = 10000,    # Learning rate ramp-up duration.
    loss_scaling        = 1,        # Loss scaling factor for reducing FP16 under/overflows.
    kimg_per_tick       = 50,       # Interval of progress prints.
    snapshot_ticks      = 50,       # How often to save network snapshots, None = disable.
    state_dump_ticks    = 500,      # How often to dump training state, None = disable.
    resume_pkl          = None,     # Start from the given network snapshot, None = random initialization.
    resume_state_dump   = None,     # Start from the given training state, None = reset training state.
    resume_kimg         = 0,        # Start from the given training progress.
    cudnn_benchmark     = True,     # Enable torch.backends.cudnn.benchmark?
    device              = torch.device('cuda'),
    # TD-related parameters
    loss_type           = 'edm',    # Loss type: 'edm' or 'td'
    lambda_edm          = 0.01,     # lambda coefficient for EDM loss in mixed objective
    target_update_freq  = 100,      # Target network update frequency (in steps)
    target_update_type  = 'hard',   # Target network update type: 'hard' or 'ema'
    target_ema_tau      = 0.999,    # EMA coefficient for target network (only used when target_update_type='ema')
    # FID eval & early stopping
    fid_eval_every      = 5,        # Evaluate FID every N ticks. None/0 = disable.
    fid_num_images      = 50000,    # Number of generated images for FID.
    fid_ref_path        = 'fid-refs/cifar10-32x32.npz',  # Reference stats path.
    fid_max_batch       = 64,       # Max batch when computing Inception features.
    early_stop_fid      = None,     # Stop if FID <= this threshold. None = disable.
    fid_nfe_grid        = (9, 15, 25, 35),  # Evaluate FID at these NFEs (steps=5,8,12,18).
):
    # Initialize.
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Select batch size per GPU.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()

    # Load dataset.
    dist.print0('Loading dataset...')
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # subclass of training.dataset.Dataset
    dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed)
    dataset_iterator = iter(torch.utils.data.DataLoader(dataset=dataset_obj, sampler=dataset_sampler, batch_size=batch_gpu, **data_loader_kwargs))

    # Construct network.
    dist.print0('Constructing network...')
    interface_kwargs = dict(img_resolution=dataset_obj.resolution, img_channels=dataset_obj.num_channels, label_dim=dataset_obj.label_dim)
    net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs) # subclass of torch.nn.Module
    net.train().requires_grad_(True).to(device)
    
    # Initialize target network for TD loss
    target_net = None
    if loss_type == 'td':
        target_net = copy.deepcopy(net)
        target_net.eval().requires_grad_(False)
        target_net.to(device)
        dist.print0('Initialized target network for TD loss')
    
    if dist.get_rank() == 0:
        with torch.no_grad():
            images = torch.zeros([batch_gpu, net.img_channels, net.img_resolution, net.img_resolution], device=device)
            sigma = torch.ones([batch_gpu], device=device)
            labels = torch.zeros([batch_gpu, net.label_dim], device=device)
            misc.print_module_summary(net, [images, sigma, labels], max_nesting=2)

    # Setup optimizer.
    dist.print0('Setting up optimizer...')
    if loss_type == 'td':
        # For TD loss, we need to pass additional parameters
        loss_kwargs_td = loss_kwargs.copy()
        loss_kwargs_td['lambda_edm'] = lambda_edm
        loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs_td)
    else:
        loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs)
    
    optimizer = dnnlib.util.construct_class_by_name(params=net.parameters(), **optimizer_kwargs) # subclass of torch.optim.Optimizer
    augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs) if augment_kwargs is not None else None # training.augment.AugmentPipe
    ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device])
    ema = copy.deepcopy(net).eval().requires_grad_(False)

    # Resume training from previous snapshot.
    if resume_pkl is not None:
        dist.print0(f'Loading network weights from "{resume_pkl}"...')
        if dist.get_rank() != 0:
            torch.distributed.barrier() # rank 0 goes first
        with dnnlib.util.open_url(resume_pkl, verbose=(dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        if dist.get_rank() == 0:
            torch.distributed.barrier() # other ranks follow
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=net, require_all=False)
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=ema, require_all=False)
        if target_net is not None:
            misc.copy_params_and_buffers(src_module=data['ema'], dst_module=target_net, require_all=False)
        del data # conserve memory
    if resume_state_dump:
        dist.print0(f'Loading training state from "{resume_state_dump}"...')
        data = torch.load(resume_state_dump, map_location=torch.device('cpu'))
        misc.copy_params_and_buffers(src_module=data['net'], dst_module=net, require_all=True)
        if target_net is not None and 'target_net' in data:
            misc.copy_params_and_buffers(src_module=data['target_net'], dst_module=target_net, require_all=True)
        optimizer.load_state_dict(data['optimizer_state'])
        del data # conserve memory

    # Train.
    dist.print0(f'Training for {total_kimg} kimg...')
    dist.print0()
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    dist.update_progress(cur_nimg // 1000, total_kimg)
    stats_jsonl = None
    step_count = 0  # Track steps for target network updates
    
    while True:

        # Accumulate gradients.
        optimizer.zero_grad(set_to_none=True)
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                images, labels = next(dataset_iterator)
                images = images.to(device).to(torch.float32) / 127.5 - 1
                labels = labels.to(device)
                
                # Compute loss based on loss type
                if loss_type == 'td':
                    loss = loss_fn(net=ddp, target_net=target_net, images=images, labels=labels, augment_pipe=augment_pipe)
                else:
                    loss = loss_fn(net=ddp, images=images, labels=labels, augment_pipe=augment_pipe)
                
                training_stats.report('Loss/loss', loss)
                loss.sum().mul(loss_scaling / batch_gpu_total).backward()

        # Update weights.
        for g in optimizer.param_groups:
            g['lr'] = optimizer_kwargs['lr'] * min(cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1)
        for param in net.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
        optimizer.step()

        # Update target network for TD loss
        if loss_type == 'td' and target_net is not None:
            step_count += 1
            if target_update_type == 'hard':
                # Hard update at specified frequency
                if step_count % target_update_freq == 0:
                    update_target_network_hard(net, target_net)
                    training_stats.report('TD/target_update_hard', 1.0)
            elif target_update_type == 'ema':
                # EMA update every step
                update_target_network_ema(net, target_net, tau=target_ema_tau)
                training_stats.report('TD/target_update_ema', target_ema_tau)

        # Update EMA.
        ema_halflife_nimg = ema_halflife_kimg * 1000
        if ema_rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
        ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
        for p_ema, p_net in zip(ema.parameters(), net.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

        # Perform maintenance tasks once per tick.
        cur_nimg += batch_size
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        if loss_type == 'td':
            fields += [f"lambda_edm {lambda_edm:<6.3f}"]
            if target_update_type == 'ema':
                fields += [f"target_tau {target_ema_tau:<6.4f}"]
            else:
                fields += [f"target_freq {target_update_freq:<4d}"]
        torch.cuda.reset_peak_memory_stats()
        dist.print0(' '.join(fields))

        # === FID eval (configurable) ===
        if fid_eval_every and fid_eval_every > 0 and cur_tick % fid_eval_every == 0 and dist.get_rank() == 0:
            try:
                # 1) Save EMA model
                ema_pkl = os.path.join(run_dir, f'network-snapshot-fid-{cur_tick:06d}.pkl')
                with open(ema_pkl, 'wb') as f:
                    pickle.dump({'ema': ema}, f)

                # 2) Evaluate FID for each requested NFE (mapped to steps)
                nfe_list = list(fid_nfe_grid) if fid_nfe_grid is not None else [35]
                steps_list = [(int(nfe) + 1) // 2 for nfe in nfe_list]

                fid_steps18_value = None

                for nfe, steps in zip(nfe_list, steps_list):
                    # 2a) Generate images to a temp dir specific to current steps
                    gen_dir = os.path.join(run_dir, f'fid-tmp-tick{cur_tick}-steps{steps}')
                    os.makedirs(gen_dir, exist_ok=True)
                    last_seed = max(int(fid_num_images) - 1, 1)
                    gen_cmd = [
                        'python', 'generate.py',
                        f'--outdir={gen_dir}', f'--seeds=0-{last_seed}', '--subdirs', f'--network={ema_pkl}',
                        f'--steps={steps}',
                    ]
                    env = os.environ.copy()
                    # Avoid MASTER_PORT collision across parallel trainings by
                    # deriving a unique port from the parent training process port.
                    # Each training in run_4way_parallel.sh already uses a distinct
                    # MASTER_PORT (e.g., 29501..29504). Offset by +100 and add
                    # tick-based jitter to avoid clashes within the same run.
                    parent_port = int(env.get('MASTER_PORT', '29500'))
                    env['MASTER_PORT'] = str(parent_port + 100 + (cur_tick % 100))
                    env['WORLD_SIZE'] = '1'
                    env['RANK'] = '0'
                    env['LOCAL_RANK'] = '0'
                    dist.print0('Running (gen): ' + ' '.join(gen_cmd) + f" [MASTER_PORT={env['MASTER_PORT']}]")
                    subprocess.run(gen_cmd, env=env, check=True)

                    # 2b) Compute FID in-process
                    mu, sigma = calculate_inception_stats(image_path=gen_dir, num_expected=fid_num_images, max_batch_size=fid_max_batch)
                    with dnnlib.util.open_url(fid_ref_path) as f:
                        ref = dict(np.load(f))
                    fid_value = calculate_fid_from_inception_stats(mu, sigma, ref['mu'], ref['sigma'])

                    # 2c) Report & log per-steps
                    training_stats.report(f'FID/value_steps{steps}', torch.as_tensor(fid_value))
                    out_path = os.path.join(run_dir, f'fid_{steps}.jsonl')
                    with open(out_path, 'a') as f:
                        f.write(json.dumps({'tick': int(cur_tick), 'kimg': float(cur_nimg/1000.0), 'fid': float(fid_value), 'steps': int(steps), 'nfe': int(nfe), 'time_sec': float(time.time() - start_time)}) + '\n')

                    if steps == 18:
                        fid_steps18_value = fid_value

                # 3) Early stopping uses steps=18 only
                if (early_stop_fid is not None) and (fid_steps18_value is not None) and (fid_steps18_value <= early_stop_fid):
                    dist.print0(f'Early stopping: FID@steps=18 {fid_steps18_value:.3f} <= target {early_stop_fid:.3f}')
                    done = True
            except Exception as e:
                dist.print0(f'FID evaluation failed: {e}')
        # === FID eval end ===

        # Check for abort.
        if (not done) and dist.should_stop():
            done = True
            dist.print0()
            dist.print0('Aborting...')

        # Save network snapshot.
        if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0):
            data = dict(ema=ema, loss_fn=loss_fn, augment_pipe=augment_pipe, dataset_kwargs=dict(dataset_kwargs))
            if target_net is not None:
                data['target_net'] = target_net
            for key, value in data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    misc.check_ddp_consistency(value)
                    data[key] = value.cpu()
                del value # conserve memory
            if dist.get_rank() == 0:
                with open(os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl'), 'wb') as f:
                    pickle.dump(data, f)
            del data # conserve memory

        # Save full dump of the training state.
        if (state_dump_ticks is not None) and (done or cur_tick % state_dump_ticks == 0) and cur_tick != 0 and dist.get_rank() == 0:
            state_dict = dict(net=net, optimizer_state=optimizer.state_dict())
            if target_net is not None:
                state_dict['target_net'] = target_net
            torch.save(state_dict, os.path.join(run_dir, f'training-state-{cur_nimg//1000:06d}.pt'))

        # Update logs.
        training_stats.default_collector.update()
        if dist.get_rank() == 0:
            if stats_jsonl is None:
                stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
            stats_jsonl.write(json.dumps(dict(training_stats.default_collector.as_dict(), timestamp=time.time())) + '\n')
            stats_jsonl.flush()
        dist.update_progress(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    dist.print0()
    dist.print0('Exiting...')

#----------------------------------------------------------------------------
