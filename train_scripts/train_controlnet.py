import os
import sys
import types
from pathlib import Path
current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))
import argparse
import datetime
import math
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")  # ignore warning
from accelerate import DistributedDataParallelKwargs
from mmcv.runner import LogBuffer
from tqdm import tqdm
from copy import deepcopy
from diffusion.utils.checkpoint import save_checkpoint, load_checkpoint

import torch
import torch.nn as nn
from PIL import Image
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import DistributedType
from diffusers.models import AutoencoderKL
from torchvision.utils import make_grid
from torch.utils.data import RandomSampler

from diffusion import IDDPM
from diffusion.utils.dist_utils import synchronize, get_world_size, clip_grad_norm_
from diffusion.data.builder import build_dataset, build_dataloader, set_data_root
from diffusion.model.builder import build_model
from diffusion.utils.logger import get_root_logger
from diffusion.utils.misc import set_random_seed, read_config, init_random_seed, DebugUnderflowOverflow #MoxingWorker
from diffusion.utils.optimizer import build_optimizer, auto_scale_lr
from diffusion.utils.lr_scheduler import build_lr_scheduler
from diffusion.utils.data_sampler import AspectRatioBatchSampler, BalancedAspectRatioBatchSampler
from diffusion.model.nets import PixArtMSBlock, PixArtMS
from diffusion.model.nets import ControlT2IDiT, ControlPixArt_Mid, ControlPixArtAll, ControlPixArtHalf

def set_fsdp_env():
    os.environ["ACCELERATE_USE_FSDP"] = 'true'
    os.environ["FSDP_AUTO_WRAP_POLICY"] = 'TRANSFORMER_BASED_WRAP'
    os.environ["FSDP_BACKWARD_PREFETCH"] = 'BACKWARD_PRE'
    os.environ["FSDP_TRANSFORMER_CLS_TO_WRAP"] = 'PixArtBlock'


def ema_update(model_dest: nn.Module, model_src: nn.Module, rate):
    param_dict_src = dict(model_src.named_parameters())
    for p_name, p_dest in model_dest.named_parameters():
        p_src = param_dict_src[p_name]
        assert p_src is not p_dest
        p_dest.data.mul_(rate).add_((1 - rate) * p_src.data)

def train():
    if config.get('debug_nan', False):
        DebugUnderflowOverflow(model)
        logger.info('NaN debugger registered. Start to detect overflow during training.')
    time_start, last_tic = time.time(), time.time()
    log_buffer = LogBuffer()

    start_step = start_epoch * len(train_dataloader)
    global_step = 0
    total_steps = len(train_dataloader) * config.num_epochs

    # Now you train the model
    for epoch in range(start_epoch + 1, config.num_epochs + 1):
        data_time_start= time.time()
        data_time_all = 0
        for step, batch in enumerate(train_dataloader):
            data_time_all += time.time() - data_time_start
            z = batch[0] # 4 x 4 x 128 x 128 z:vae output, 3x1024x1024->vae->4x128x128
            clean_images = z * config.scale_factor #vae needed scale factor
            y = batch[1] # 4 x 1 x 120 x 4096 # T5 extracted feature of caption, 120 token, 4096
            y_mask = batch[2] # 4 x 1 x 1 x 120 # caption indicate whether valid
            data_info = batch[3]
            # data_info contains img_hw, aspect_ratio, and mask(useless) and condition 
            # condition shape is 4 x 4 x 128 x 128
            

            # Sample a random timestep for each image
            bs = clean_images.shape[0]
            timesteps = torch.randint(0, config.train_sampling_steps, (bs,), device=clean_images.device).long()
            grad_norm = None
            with accelerator.accumulate(model):
                # Predict the noise residual
                optimizer.zero_grad()
                loss_term = train_diffusion.training_losses(model, clean_images, timesteps, model_kwargs=dict(y=y, mask=y_mask, data_info=data_info, c=data_info['condition'] * config.scale_factor))
                loss = loss_term['loss'].mean()
                accelerator.backward(loss)
                # for n, p in model.named_parameters():
                #     print(n, '\'s grad is None' if p.grad is None else f'\'s grad is not None {p.max()} {p.min()}')
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), config.gradient_clip)
                optimizer.step()
                lr_scheduler.step()
                if accelerator.sync_gradients:
                    ema_update(model_ema, model, config.ema_rate)

            lr = lr_scheduler.get_last_lr()[0]
            logs = {"loss": accelerator.gather(loss).mean().item()}
            if grad_norm is not None:
                logs.update(grad_norm=accelerator.gather(grad_norm).mean().item())
            log_buffer.update(logs)
            if (step + 1) % config.log_interval == 0 or (step + 1) == 1:
                t = (time.time() - last_tic) / config.log_interval
                t_d = data_time_all / config.log_interval
                avg_time = (time.time() - time_start) / (global_step + 1)
                eta = str(datetime.timedelta(seconds=int(avg_time * (total_steps - start_step - global_step - 1))))
                eta_epoch = str(datetime.timedelta(seconds=int(avg_time * (len(train_dataloader) - step - 1))))
                # avg_loss = sum(loss_buffer) / len(loss_buffer)
                log_buffer.average()
                info = f"Step/Epoch [{(epoch-1)*len(train_dataloader)+step+1}/{epoch}][{step + 1}/{len(train_dataloader)}]:total_eta: {eta}, " \
                       f"epoch_eta:{eta_epoch}, time_all:{t:.3f}, time_data:{t_d:.3f}, lr:{lr:.3e}, s:({model.module.h}, {model.module.w}), "
                info += ', '.join([f"{k}:{v:.4f}" for k, v in log_buffer.output.items()])
                logger.info(info)
                last_tic = time.time()
                log_buffer.clear()
                data_time_all = 0
            logs.update(lr=lr)
            accelerator.log(logs, step=global_step + start_step)

            # moxing tensorboard log to s3
            # if (global_step + 1) % config.tensorboard_mox_interval == 0 and config.s3_work_dir is not None:
            #     mox_worker.mox(os.path.join(config.work_dir, 'logs'),
            #                    os.path.join(config.s3_work_dir, 'logs'))
            if (global_step + 1) % 1000 == 0 and config.s3_work_dir is not None:
                logger.info(f"s3_work_dir: {config.s3_work_dir}")

            global_step += 1
            data_time_start= time.time()

            synchronize()
            # After each epoch you optionally sample some demo images with evaluate() and save the model
            if accelerator.is_main_process:
                if ((epoch - 1) * len(train_dataloader) + step + 1) % config.save_model_steps == 0:
                    os.umask(0o000)  # file permission: 666; dir permission: 777
                    save_checkpoint(os.path.join(config.work_dir, 'checkpoints'),
                                    epoch=epoch,
                                    step=(epoch - 1) * len(train_dataloader) + step + 1,
                                    model=accelerator.unwrap_model(model),
                                    model_ema=accelerator.unwrap_model(model_ema),
                                    optimizer=optimizer,
                                    lr_scheduler=lr_scheduler
                                    )
            synchronize()

        synchronize()
        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            if epoch % config.save_model_epochs == 0 or epoch == config.num_epochs:
                os.umask(0o000)  # file permission: 666; dir permission: 777
                save_checkpoint(os.path.join(config.work_dir, 'checkpoints'),
                                epoch=epoch,
                                step=(epoch - 1) * len(train_dataloader) + step + 1,
                                model=accelerator.unwrap_model(model),
                                model_ema=accelerator.unwrap_model(model_ema),
                                optimizer=optimizer,
                                lr_scheduler=lr_scheduler
                                )
        synchronize()


def parse_args():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("config", type=str, help="config")
    parser.add_argument("--cloud", action='store_true', default=False, help="cloud or local machine")
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--resume_from', help='the dir to save logs and models')
    parser.add_argument('--local-rank', type=int, default=-1)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--debug', action='store_true')

    parser.add_argument('--save_step', type=int, default=400)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--data_root', type=str, default=None)
    parser.add_argument('--resume_optimizer', action='store_true')
    parser.add_argument('--resume_lr_scheduler', action='store_true')
    parser.add_argument('--controlnet_type', type=str, default='all', \
        help='the network architecture of controlnet, choose from all or half')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    config = read_config(args.config)
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        config.work_dir = args.work_dir
    if args.cloud:
        config.data_root = '/data/data'
    if args.data_root:
        config.data_root = args.data_root
    if args.resume_from is not None:
        resume_dict = dict(
            checkpoint=args.resume_from,
            load_ema=False,
            resume_optimizer=args.resume_optimizer,
            resume_lr_scheduler=args.resume_lr_scheduler)
    if args.debug:
        config.log_interval = 1
        config.train_batch_size = 2
        config.save_model_steps=args.save_step
        config.optimizer.update({'lr': args.lr})

    os.umask(0o000)  # file permission: 666; dir permission: 777
    os.makedirs(config.work_dir, exist_ok=True)
    # mox_worker = MoxingWorker()
    # mox_worker.start()

    init_handler = InitProcessGroupKwargs()
    init_handler.timeout = datetime.timedelta(seconds=9600)  # change timeout to avoid a strange NCCL bug
    # Initialize accelerator and tensorboard logging
    if config.use_fsdp:
        init_train = 'FSDP'
        from accelerate import FullyShardedDataParallelPlugin
        from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig
        set_fsdp_env()
        fsdp_plugin = FullyShardedDataParallelPlugin(state_dict_config=FullStateDictConfig(offload_to_cpu=False, rank0_only=False),)
    else:
        init_train = 'DDP'
        fsdp_plugin = None

    even_batches = True
    if config.multi_scale:
        even_batches=False,


    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.work_dir, "logs"),
        fsdp_plugin=fsdp_plugin,
        even_batches=even_batches,
        kwargs_handlers=[init_handler]
    )

    logger = get_root_logger(os.path.join(config.work_dir, 'train_log.log'))

    config.seed = init_random_seed(config.get('seed', None))
    set_random_seed(config.seed)

    if accelerator.is_main_process:
        config.dump(os.path.join(config.work_dir, 'config.py'))

    logger.info(f"Config: \n{config.pretty_text}")
    logger.info(f"World_size: {get_world_size()}, seed: {config.seed}")
    logger.info(f"Initializing: {init_train} for training")
    image_size = config.image_size  # @param [256, 512]
    latent_size = int(image_size) // 8
    pred_sigma = getattr(config, 'pred_sigma', True)
    learn_sigma = getattr(config, 'learn_sigma', True) and pred_sigma
    model_kwargs={"window_block_indexes": config.window_block_indexes, "window_size": config.window_size,
                  "use_rel_pos": config.use_rel_pos, "lewei_scale": config.lewei_scale, 'config':config}

    # build models
    train_diffusion = IDDPM(str(config.train_sampling_steps))
    eval_diffusion = IDDPM(str(config.eval_sampling_steps))

    model: PixArtMS = build_model(config.model,
                        config.grad_checkpointing,
                        config.get('fp32_attention', False),
                        input_size=latent_size,
                        learn_sigma=learn_sigma,
                        pred_sigma=pred_sigma,
                        **model_kwargs)

    if config.load_from is not None and args.resume_from is None:
        # load from pixart model
        missing, unexpected = load_checkpoint(config.load_from, model, load_ema=config.get('load_ema', False))
        # model.reparametrize()
        if accelerator.is_main_process:
            print('Warning Missing keys: ', missing)
            print('Warning Unexpected keys', unexpected)

    if args.controlnet_type == 'all':
        model = ControlPixArtAll(model)
    else:
        model = ControlPixArtHalf(model)

    model = model.train()
    logger.info(f"{model.__class__.__name__} Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    if args.local_rank == 0:
        for name, params in model.named_parameters():
            if params.requires_grad == False: logger.info(f"freeze param: {name}")

        for name, params in model.named_parameters():
            if params.requires_grad == True: logger.info(f"trainable param: {name}")

    model_ema = deepcopy(model).eval()
    ema_update(model_ema, model, 0.)

    # prepare for FSDP clip grad norm calculation
    if accelerator.distributed_type == DistributedType.FSDP:
        for m in accelerator._models:
            m.clip_grad_norm_ = types.MethodType(clip_grad_norm_, m)

    # build dataloader
    set_data_root(config.data_root)
    dataset = build_dataset(config.data, resolution=image_size, aspect_ratio_type=config.aspect_ratio_type, train_ratio=config.train_ratio)
    if config.multi_scale:
        batch_sampler = AspectRatioBatchSampler(sampler=RandomSampler(dataset), dataset=dataset,
                                                batch_size=config.train_batch_size, aspect_ratios=dataset.aspect_ratio, drop_last=True,
                                                ratio_nums=dataset.ratio_nums, config=config, valid_num=1)
        # batch_sampler = BalancedAspectRatioBatchSampler(sampler=RandomSampler(dataset), dataset=dataset,
        #                                                 batch_size=config.train_batch_size, aspect_ratios=dataset.aspect_ratio,
        #                                                 ratio_nums=dataset.ratio_nums)
        train_dataloader = build_dataloader(dataset, batch_sampler=batch_sampler, num_workers=config.num_workers)
    else:
        train_dataloader = build_dataloader(dataset, num_workers=config.num_workers, batch_size=config.train_batch_size, shuffle=True)
    # print("debug train_dataloader", train_dataloader)

    # build optimizer and lr scheduler
    lr_scale_ratio = 1
    if config.get('auto_lr', None):
        lr_scale_ratio = auto_scale_lr(config.train_batch_size * get_world_size() * config.gradient_accumulation_steps,
                                       config.optimizer,
                                       **config.auto_lr)
    # optimizer = build_optimizer(model, config.optimizer)
    optimizer = build_optimizer(model.controlnet, config.optimizer)
    lr_scheduler = build_lr_scheduler(config, optimizer, train_dataloader, lr_scale_ratio)

    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())

    if accelerator.is_main_process:
        accelerator.init_trackers(f"tb_{timestamp}")

    start_epoch = 0
    if args.resume_from is not None:
        if args.resume_optimizer == False or args.resume_lr_scheduler == False:
            missing, unexpected = load_checkpoint(args.resume_from, model, load_ema=config.get('load_ema', False))
        else:
            start_epoch, missing, unexpected = load_checkpoint(**resume_dict,
                                                           model=model,
                                                           model_ema=model_ema,
                                                           optimizer=optimizer,
                                                           lr_scheduler=lr_scheduler,
                                                           )

        if accelerator.is_main_process:
            print('Warning Missing keys: ', missing)
            print('Warning Unexpected keys', unexpected)
    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, model_ema = accelerator.prepare(model, model_ema)
    optimizer, train_dataloader, lr_scheduler = accelerator.prepare(optimizer, train_dataloader, lr_scheduler)
    train()
    # mox_worker.close()
