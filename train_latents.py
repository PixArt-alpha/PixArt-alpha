import os

import sys
import types
from pathlib import Path
current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))
import argparse
import datetime
import time
import warnings
warnings.filterwarnings("ignore")  # ignore warning
import torch
import torch.nn as nn
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import DistributedType
from diffusers.models import AutoencoderKL
from torch.utils.data import RandomSampler
from mmcv.runner import LogBuffer
from copy import deepcopy
from PIL import Image
import numpy as np

from diffusion import IDDPM
from diffusion.utils.checkpoint import save_checkpoint, load_checkpoint
from diffusion.utils.dist_utils import synchronize, get_world_size, clip_grad_norm_
from diffusion.data.builder import build_dataset, build_dataloader, set_data_root
from diffusion.model.builder import build_model
from diffusion.utils.logger import get_root_logger
from diffusion.utils.misc import set_random_seed, read_config, init_random_seed, DebugUnderflowOverflow
from diffusion.utils.optimizer import build_optimizer, auto_scale_lr
from diffusion.utils.lr_scheduler import build_lr_scheduler
from diffusion.utils.data_sampler import AspectRatioBatchSampler, BalancedAspectRatioBatchSampler

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

    # load_vae_feat = getattr(train_dataloader.dataset, 'load_vae_feat', False)
    # Now you train the model
    for epoch in range(start_epoch + 1, config.num_epochs + 1):
        data_time_start= time.time()
        data_time_all = 0
        for step, batch in enumerate(train_dataloader):
            data_time_all += time.time() - data_time_start
            # if load_vae_feat:
            z = batch[0]
            # else:
            #     with torch.no_grad():
            #         with torch.cuda.amp.autocast(enabled=config.mixed_precision == 'fp16'):
            #             posterior = vae.encode(batch[0]).latent_dist
            #             if config.sample_posterior:
            #                 z = posterior.sample()
            #             else:
            #                 z = posterior.mode()
            clean_images = z * config.scale_factor
            y = batch[1]
            y_mask = batch[2]
            data_info = batch[3]

            # Sample a random timestep for each image
            bs = clean_images.shape[0]
            timesteps = torch.randint(0, config.train_sampling_steps, (bs,), device=clean_images.device).long()
            grad_norm = None
            with accelerator.accumulate(model):
                # Predict the noise residual
                optimizer.zero_grad()
                loss_term = train_diffusion.training_losses(model, clean_images, timesteps, model_kwargs=dict(y=y, mask=y_mask, data_info=data_info))
                loss = loss_term['loss'].mean()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), config.gradient_clip)
                optimizer.step()
                lr_scheduler.step()
                if accelerator.sync_gradients:
                    ema_update(model_ema, model, config.ema_rate)

            lr = lr_scheduler.get_last_lr()[0]
            logs = {args.loss_report_name: accelerator.gather(loss).mean().item()}
            if grad_norm is not None:
                logs.update(grad_norm=accelerator.gather(grad_norm).mean().item())
            log_buffer.update(logs)
            
            # logging on terminal
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

            global_step += 1
            data_time_start= time.time()

            synchronize()
            if accelerator.is_main_process:
                if ((epoch - 1) * len(train_dataloader) + step + 1) % config.save_model_steps == 0:
                    os.umask(0o000)
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
        if accelerator.is_main_process:
            if epoch % config.save_model_epochs == 0 or epoch == config.num_epochs:
                os.umask(0o000)
                save_checkpoint(os.path.join(config.output_dir, 'checkpoints'),
                                epoch=epoch,
                                step=(epoch - 1) * len(train_dataloader) + step + 1,
                                model=accelerator.unwrap_model(model),
                                model_ema=accelerator.unwrap_model(model_ema),
                                optimizer=optimizer,
                                lr_scheduler=lr_scheduler
                                )
            ########### EVAL ###################
            if epoch % config.save_image_epochs == 0 or epoch == config.num_epochs:                
                if config.validation_prompts is not None:
                    logger.info("Running inference for collecting generated images...")
      
                    assert config.eval_sampler in ['iddpm', 'dpm-solver', 'sa-solver']
                    sample_steps_dict = {'iddpm': 100, 'dpm-solver': 20, 'sa-solver': 25}
                    sample_steps = config.eval_steps if config.eval_steps != -1 else sample_steps_dict[config.eval_sampler]
                    # base_ratios = eval(f'ASPECT_RATIO_{config.image_size}_TEST')
                    
                    eval_dir = os.path.join(config.output_dir, 'eval')
                    os.makedirs(eval_dir, exist_ok=True)
                    save_path = os.path.join(eval_dir, f'{epoch}_{global_step}.png')
                    
                    model.eval()
                    images = []
                    # device = t5.device
                    for ip, prompt in enumerate(config.validation_prompts):
                        prompts = [prompt]
                        # prompts = []
                        # prompt_clean, _, hw, ar, custom_hw = prepare_prompt_ar(prompt, base_ratios, device=device, show=False)  # ar for aspect ratio
                        # if config.image_size == 1024:
                            # latent_size_h, latent_size_w = int(hw[0, 0] // 8), int(hw[0, 1] // 8)
                        # else:
                        #     hw = torch.tensor([[config.image_size, config.image_size]], dtype=torch.float, device=device).repeat(bs, 1)
                        #     ar = torch.tensor([[1.]], device=device).repeat(bs, 1)
                        #     latent_size_h, latent_size_w = latent_size, latent_size
                        # prompts.append(prompt_clean.strip())
                        null_y = model.module.y_embedder.y_embedding[None].repeat(len(prompts), 1, 1)[:, None]
                        
                        with torch.no_grad():
                            caption_embs, emb_masks, len_prompts = val_txt_embs[ip]
                            # caption_embs, emb_masks = t5.get_text_embeddings(prompts)
                            # caption_embs = caption_embs.float()[:, None]
                            print(f'finish embedding')
                            n = len_prompts
                            if config.eval_sampler == 'iddpm':
                                # Create sampling noise:
                                z = torch.randn(n, 4, latent_size_h, latent_size_w, device=device).repeat(2, 1, 1, 1)
                                model_kwargs = dict(y=torch.cat([caption_embs, null_y]),
                                                    cfg_scale=config.cfg_scale, data_info={'img_hw': hw, 'aspect_ratio': ar}, mask=emb_masks)
                                diffusion = IDDPM(str(sample_steps))
                                # Sample images:
                                samples = diffusion.p_sample_loop(
                                    model.module.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True,
                                    device=device
                                )
                                samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
                            elif config.eval_sampler == 'dpm-solver':
                                # Create sampling noise:
                                z = torch.randn(n, 4, latent_size_h, latent_size_w, device=device)
                                model_kwargs = dict(data_info={'img_hw': hw, 'aspect_ratio': ar}, mask=emb_masks)
                                dpm_solver = DPMS(model.module.forward_with_dpmsolver,
                                                condition=caption_embs,
                                                uncondition=null_y,
                                                cfg_scale=config.cfg_scale,
                                                model_kwargs=model_kwargs)
                                samples = dpm_solver.sample(
                                    z,
                                    steps=sample_steps,
                                    order=2,
                                    skip_type="time_uniform",
                                    method="multistep",
                                )
                            elif config.eval_sampler == 'sa-solver':
                                # Create sampling noise:
                                model_kwargs = dict(data_info={'img_hw': hw, 'aspect_ratio': ar}, mask=emb_masks)
                                sa_solver = SASolverSampler(model.module.forward_with_dpmsolver, device=device)
                                samples = sa_solver.sample(
                                    S=25,
                                    batch_size=n,
                                    shape=(4, latent_size_h, latent_size_w),
                                    eta=1,
                                    conditioning=caption_embs,
                                    unconditional_conditioning=null_y,
                                    unconditional_guidance_scale=config.cfg_scale,
                                    model_kwargs=model_kwargs,
                                )[0]
                        samples = vae.decode(samples / 0.18215).sample
                        # decode image
                        image = make_grid(samples, nrow=1, normalize=True, value_range=(-1, 1))
                        image = image.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
                        image = Image.fromarray(image)
                        images.append(image)
                        
                    image_grid = make_image_grid(images, 2, len(images)//2)
                    image_grid.save(save_path)
                    for tracker in accelerator.trackers:
                        if tracker.name == "tensorboard":
                            np_images = np.stack([np.asarray(img) for img in images])
                            tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
                        elif tracker.name == "comet_ml":
                            logger.info('Logging validation images')
                            tracker.writer.log_image(image_grid, name=f"{epoch}", step=global_step)
                        else:
                            logger.warn(f"image logging not implemented for {tracker.name}")
                    
                    del images, image, samples, image_grid
                    torch.cuda.empty_cache()
                        
        model.train()
        synchronize()


def parse_args():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("config", type=str, help="config")
    parser.add_argument("--cloud", action='store_true', default=False, help="cloud or local machine")
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--resume-from', help='the dir to resume the training')
    parser.add_argument('--load-from', default=None, help='the dir to load a ckpt for training')
    parser.add_argument('--local-rank', type=int, default=-1)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument("--loss_report_name", type=str, default="loss")
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
    if args.resume_from is not None:
        config.load_from = None
        config.resume_from = dict(
            checkpoint=args.resume_from,
            load_ema=False,
            resume_optimizer=True,
            resume_lr_scheduler=True)
    if args.debug:
        config.log_interval = 1
        config.train_batch_size = 8
        config.valid_num = 100

    os.umask(0o000)
    config.output_dir = os.path.join(config.work_dir, 
                                     f"""{config.model}_{config.dataset_alias}_{config.image_size}_batch{config.train_batch_size}_{config.lr_schedule}_lr{config.optimizer['lr']}_warmup{config.lr_schedule_args['num_warmup_steps']}_gas{config.gradient_accumulation_steps}""")        
    os.makedirs(config.output_dir, exist_ok=True)

    init_handler = InitProcessGroupKwargs()
    init_handler.timeout = datetime.timedelta(seconds=5400)  # change timeout to avoid a strange NCCL bug
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
        
    if args.report_to == "comet_ml":
        import comet_ml
        comet_ml.init(
            project_name=args.tracker_project_name,
        )     

    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with=args.report_to,
        project_dir=os.path.join(config.output_dir, "logs"),
        fsdp_plugin=fsdp_plugin,
        even_batches=even_batches,
        kwargs_handlers=[init_handler]
    )

    logger = get_root_logger(os.path.join(config.output_dir, 'train_log.log'))

    config.seed = init_random_seed(config.get('seed', None))
    set_random_seed(config.seed)

    if accelerator.is_main_process:
        config.dump(os.path.join(config.output_dir, 'config.py'))

    logger.info(f"Config: \n{config.pretty_text}")
    logger.info(f"World_size: {get_world_size()}, seed: {config.seed}")
    logger.info(f"Initializing: {init_train} for training")
    image_size = config.image_size  # @param [256, 512]
    latent_size = int(image_size) // 8
    pred_sigma = getattr(config, 'pred_sigma', True)
    learn_sigma = getattr(config, 'learn_sigma', True) and pred_sigma
    model_kwargs={"window_block_indexes": config.window_block_indexes, "window_size": config.window_size,
                  "use_rel_pos": config.use_rel_pos, "lewei_scale": config.lewei_scale, 'config':config,
                  'model_max_length': config.model_max_length}
    
    if config.validation_prompts is not None:
        logger.info('Precompute validation prompt embeddings')
        from diffusion.model.utils import prepare_prompt_ar
        from diffusion import IDDPM, DPMS, SASolverSampler
        from diffusion.model.t5 import T5Embedder
        from diffusion.data.datasets import ASPECT_RATIO_256_TEST, ASPECT_RATIO_512_TEST, ASPECT_RATIO_1024_TEST
        from diffusers.utils import  make_image_grid
        from torchvision.utils import make_grid
        
        t5 = T5Embedder(device="cuda", local_cache=True, cache_dir='output/pretrained_models/t5_ckpts', torch_dtype=torch.float)
        device = t5.device
        base_ratios = eval(f'ASPECT_RATIO_{config.image_size}_TEST')
        pbs = 1
        val_txt_embs = []
        for prompt in config.validation_prompts:
            prompts = []
            prompt_clean, _, hw, ar, custom_hw = prepare_prompt_ar(prompt, base_ratios, device=device, show=False)  # ar for aspect ratio
            if config.image_size == 1024:
                latent_size_h, latent_size_w = int(hw[0, 0] // 8), int(hw[0, 1] // 8)
            else:
                hw = torch.tensor([[config.image_size, config.image_size]], dtype=torch.float, device=device).repeat(pbs, 1)
                ar = torch.tensor([[1.]], device=device).repeat(pbs, 1)
                latent_size_h, latent_size_w = latent_size, latent_size
            prompts.append(prompt_clean.strip())
            
            with torch.no_grad():
                caption_embs, emb_masks = t5.get_text_embeddings(prompts)
                caption_embs = caption_embs.float()[:, None]
                val_txt_embs.append([caption_embs, emb_masks, len(prompts)])
        del t5
        import gc         # garbage collect library
        gc.collect()
        torch.cuda.empty_cache()
        logger.info('[ DONE ]')

    # build models
    train_diffusion = IDDPM(str(config.train_sampling_steps), learn_sigma=learn_sigma, pred_sigma=pred_sigma, snr=config.snr_loss)
    model = build_model(config.model,
                        config.grad_checkpointing,
                        config.get('fp32_attention', False),
                        input_size=latent_size,
                        learn_sigma=learn_sigma,
                        pred_sigma=pred_sigma,
                        **model_kwargs).train()
    logger.info(f"{model.__class__.__name__} Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"T5 max token length: {config.model_max_length}")
    model_ema = deepcopy(model).eval()

    if config.load_from is not None:
        if args.load_from is not None:
            config.load_from = args.load_from
        missing, unexpected = load_checkpoint(config.load_from, model, load_ema=config.get('load_ema', False))
        logger.warning(f'Missing keys: {missing}')
        logger.warning(f'Unexpected keys: {unexpected}')

    ema_update(model_ema, model, 0.)
    if not config.data.load_vae_feat:
        vae = AutoencoderKL.from_pretrained(config.vae_pretrained).cuda()

    # prepare for FSDP clip grad norm calculation
    if accelerator.distributed_type == DistributedType.FSDP:
        for m in accelerator._models:
            m.clip_grad_norm_ = types.MethodType(clip_grad_norm_, m)

    # build dataloader
    set_data_root(config.data_root)
    dataset = build_dataset(config.data, resolution=image_size, aspect_ratio_type=config.aspect_ratio_type)
    if config.multi_scale:
        batch_sampler = AspectRatioBatchSampler(sampler=RandomSampler(dataset), dataset=dataset,
                                                batch_size=config.train_batch_size, aspect_ratios=dataset.aspect_ratio, drop_last=True,
                                                ratio_nums=dataset.ratio_nums, config=config, valid_num=config.valid_num)
        # used for balanced sampling
        # batch_sampler = BalancedAspectRatioBatchSampler(sampler=RandomSampler(dataset), dataset=dataset,
        #                                                 batch_size=config.train_batch_size, aspect_ratios=dataset.aspect_ratio,
        #                                                 ratio_nums=dataset.ratio_nums)
        train_dataloader = build_dataloader(dataset, batch_sampler=batch_sampler, num_workers=config.num_workers)
    else:
        logger.info(f'Batch size {config.train_batch_size}')
        train_dataloader = build_dataloader(dataset, num_workers=config.num_workers, batch_size=config.train_batch_size, shuffle=True)

    # build optimizer and lr scheduler
    lr_scale_ratio = 1
    if config.get('auto_lr', None):
        lr_scale_ratio = auto_scale_lr(config.train_batch_size * get_world_size() * config.gradient_accumulation_steps,
                                       config.optimizer, **config.auto_lr)
    optimizer = build_optimizer(model, config.optimizer)
    lr_scheduler = build_lr_scheduler(config, optimizer, train_dataloader, lr_scale_ratio)

    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())

    if accelerator.is_main_process:
        tracker_config = dict(vars(config))
        accelerator.init_trackers(args.tracker_project_name, tracker_config)
        accelerator.get_tracker("comet_ml").writer.add_tags([config.model, 
                                                            config.dataset_alias, 
                                                            config.image_size, 
                                                            config.lr_schedule, 
                                                            f'bs{config.train_batch_size}',
                                                            f'gs{config.gradient_accumulation_steps}'
                                                            ])

    start_epoch = 0
    if config.resume_from is not None and config.resume_from['checkpoint'] is not None:
        start_epoch, missing, unexpected = load_checkpoint(**config.resume_from,
                                                           model=model,
                                                           model_ema=model_ema,
                                                           optimizer=optimizer,
                                                           lr_scheduler=lr_scheduler,
                                                           )

        logger.warning(f'Missing keys: {missing}')
        logger.warning(f'Unexpected keys: {unexpected}')
    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, model_ema = accelerator.prepare(model, model_ema)
    optimizer, train_dataloader, lr_scheduler = accelerator.prepare(optimizer, train_dataloader, lr_scheduler)
    train()
