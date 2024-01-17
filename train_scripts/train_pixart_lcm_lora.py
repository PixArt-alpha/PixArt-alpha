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
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import DistributedType
from torch.utils.data import RandomSampler
from mmcv.runner import LogBuffer
import torch.nn.functional as F
import numpy as np
import re
from packaging import version
import accelerate

from diffusion import IDDPM
from diffusion.utils.dist_utils import get_world_size, clip_grad_norm_
from diffusion.data.builder import build_dataset, build_dataloader, set_data_root
from diffusion.utils.logger import get_root_logger
from diffusion.utils.misc import set_random_seed, read_config, init_random_seed, DebugUnderflowOverflow
from diffusion.utils.optimizer import build_optimizer, auto_scale_lr
from diffusion.utils.lr_scheduler import build_lr_scheduler
from diffusion.utils.data_sampler import AspectRatioBatchSampler, BalancedAspectRatioBatchSampler
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from diffusers import AutoencoderKL, Transformer2DModel, StableDiffusionPipeline, PixArtAlphaPipeline


def set_fsdp_env():
    os.environ["ACCELERATE_USE_FSDP"] = 'true'
    os.environ["FSDP_AUTO_WRAP_POLICY"] = 'TRANSFORMER_BASED_WRAP'
    os.environ["FSDP_BACKWARD_PREFETCH"] = 'BACKWARD_PRE'
    os.environ["FSDP_TRANSFORMER_CLS_TO_WRAP"] = 'PixArtBlock'

def filter_keys(key_set):
    def _f(dictionary):
        return {k: v for k, v in dictionary.items() if k in key_set}

    return _f


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]


# From LCMScheduler.get_scalings_for_boundary_condition_discrete
def scalings_for_boundary_conditions(timestep, sigma_data=0.5, timestep_scaling=10.0):
    c_skip = sigma_data**2 / ((timestep / 0.1) ** 2 + sigma_data**2)
    c_out = (timestep / 0.1) / ((timestep / 0.1) ** 2 + sigma_data**2) ** 0.5
    return c_skip, c_out


# Compare LCMScheduler.step, Step 4
def predicted_origin(model_output, timesteps, sample, prediction_type, alphas, sigmas):
    if prediction_type == "epsilon":
        sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
        alphas = extract_into_tensor(alphas, timesteps, sample.shape)
        pred_x_0 = (sample - sigmas * model_output) / alphas
    elif prediction_type == "v_prediction":
        sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
        alphas = extract_into_tensor(alphas, timesteps, sample.shape)
        pred_x_0 = alphas * sample - sigmas * model_output
    else:
        raise ValueError(f"Prediction type {prediction_type} currently not supported.")

    return pred_x_0


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class DDIMSolver:
    def __init__(self, alpha_cumprods, timesteps=1000, ddim_timesteps=50):
        # DDIM sampling parameters
        step_ratio = timesteps // ddim_timesteps

        self.ddim_timesteps = (np.arange(1, ddim_timesteps + 1) * step_ratio).round().astype(np.int64) - 1
        self.ddim_alpha_cumprods = alpha_cumprods[self.ddim_timesteps]
        self.ddim_alpha_cumprods_prev = np.asarray(
            [alpha_cumprods[0]] + alpha_cumprods[self.ddim_timesteps[:-1]].tolist()
        )
        # convert to torch tensors
        self.ddim_timesteps = torch.from_numpy(self.ddim_timesteps).long()
        self.ddim_alpha_cumprods = torch.from_numpy(self.ddim_alpha_cumprods)
        self.ddim_alpha_cumprods_prev = torch.from_numpy(self.ddim_alpha_cumprods_prev)

    def to(self, device):
        self.ddim_timesteps = self.ddim_timesteps.to(device)
        self.ddim_alpha_cumprods = self.ddim_alpha_cumprods.to(device)
        self.ddim_alpha_cumprods_prev = self.ddim_alpha_cumprods_prev.to(device)
        return self

    def ddim_step(self, pred_x0, pred_noise, timestep_index):
        alpha_cumprod_prev = extract_into_tensor(self.ddim_alpha_cumprods_prev, timestep_index, pred_x0.shape)
        dir_xt = (1.0 - alpha_cumprod_prev).sqrt() * pred_noise
        x_prev = alpha_cumprod_prev.sqrt() * pred_x0 + dir_xt
        return x_prev


def train(model):
    if config.get('debug_nan', False):
        DebugUnderflowOverflow(model)
        logger.info('NaN debugger registered. Start to detect overflow during training.')
    time_start, last_tic = time.time(), time.time()
    log_buffer = LogBuffer()

    global_step = start_step

    load_vae_feat = getattr(train_dataloader.dataset, 'load_vae_feat', False)

    # Create uncond embeds for classifier free guidance
    uncond_prompt_embeds = torch.load('output/pretrained_models/null_embed.pth', map_location='cpu').to(accelerator.device).repeat(config.train_batch_size, 1, 1, 1)

    # Now you train the model
    for epoch in range(start_epoch + 1, config.num_epochs + 1):
        data_time_start= time.time()
        data_time_all = 0
        for step, batch in enumerate(train_dataloader):
            data_time_all += time.time() - data_time_start
            if load_vae_feat:
                z = batch[0]
            else:
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=config.mixed_precision == 'fp16'):
                        posterior = vae.encode(batch[0]).latent_dist
                        if config.sample_posterior:
                            z = posterior.sample()
                        else:
                            z = posterior.mode()
            latents = (z * config.scale_factor).to(weight_dtype)
            y = batch[1].squeeze(1).to(weight_dtype)
            y_mask = batch[2].squeeze(1).squeeze(1).to(weight_dtype)
            data_info = {'resolution': batch[3]['img_hw'].to(weight_dtype), 'aspect_ratio': batch[3]['aspect_ratio'].to(weight_dtype),}

            # Sample a random timestep for each image
            grad_norm = None
            with accelerator.accumulate(model):
                # Predict the noise residual
                optimizer.zero_grad()
                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                # Sample a random timestep for each image t_n ~ U[0, N - k - 1] without bias.
                topk = config.train_sampling_steps // config.num_ddim_timesteps
                index = torch.randint(0, config.num_ddim_timesteps, (bsz,), device=latents.device).long()
                start_timesteps = solver.ddim_timesteps[index]
                timesteps = start_timesteps - topk
                timesteps = torch.where(timesteps < 0, torch.zeros_like(timesteps), timesteps)

                # Get boundary scalings for start_timesteps and (end) timesteps.
                c_skip_start, c_out_start = scalings_for_boundary_conditions(start_timesteps)
                c_skip_start, c_out_start = [append_dims(x, latents.ndim) for x in [c_skip_start, c_out_start]]
                c_skip, c_out = scalings_for_boundary_conditions(timesteps)
                c_skip, c_out = [append_dims(x, latents.ndim) for x in [c_skip, c_out]]

                # Sample a random guidance scale w from U[w_min, w_max] and embed it
                # w = (config.w_max - config.w_min) * torch.rand((bsz,)) + config.w_min
                w = config.cfg_scale * torch.ones((bsz,))
                w = w.reshape(bsz, 1, 1, 1)
                w = w.to(device=latents.device, dtype=latents.dtype)

                # Get online LCM prediction on z_{t_{n + k}}, w, c, t_{n + k}
                _, pred_x_0, noisy_model_input  = train_diffusion.training_losses_diffusers(
                    model, latents, start_timesteps,
                    model_kwargs=dict(encoder_hidden_states=y, encoder_attention_mask=y_mask, added_cond_kwargs=data_info),
                    noise=noise
                )
                model_pred = c_skip_start * noisy_model_input + c_out_start * pred_x_0

                with torch.no_grad():
                    with torch.autocast("cuda"):
                        cond_teacher_output, cond_pred_x0, _ = train_diffusion.training_losses_diffusers(
                            model_teacher, latents, start_timesteps,
                            model_kwargs=dict(encoder_hidden_states=y, encoder_attention_mask=y_mask, added_cond_kwargs=data_info),
                            noise=noise
                        )
                        # Get teacher model prediction on noisy_latents and unconditional embedding
                        uncond_teacher_output, uncond_pred_x0, _ = train_diffusion.training_losses_diffusers(
                            model_teacher, latents, start_timesteps,
                            model_kwargs=dict(encoder_hidden_states=uncond_prompt_embeds, encoder_attention_mask=y_mask, added_cond_kwargs=data_info),
                            noise=noise
                        )

                        # Perform "CFG" to get x_prev estimate (using the LCM paper's CFG formulation)
                        pred_x0 = cond_pred_x0 + w * (cond_pred_x0 - uncond_pred_x0)
                        pred_noise = cond_teacher_output + w * (cond_teacher_output - uncond_teacher_output)
                        x_prev = solver.ddim_step(pred_x0, pred_noise, index)

                # Get target LCM prediction on x_prev, w, c, t_n
                with torch.no_grad():
                    with torch.autocast("cuda", enabled=True):
                        _, pred_x_0, _ = train_diffusion.training_losses_diffusers(
                            model, x_prev.float(), timesteps,
                            model_kwargs=dict(encoder_hidden_states=y, encoder_attention_mask=y_mask, added_cond_kwargs=data_info),
                            skip_noise=True
                        )

                    target = c_skip * x_prev + c_out * pred_x_0

                # Calculate loss
                if config.loss_type == "l2":
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                elif config.loss_type == "huber":
                    loss = torch.mean(torch.sqrt((model_pred.float() - target.float()) ** 2 + config.huber_c**2) - config.huber_c)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), config.gradient_clip)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

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
                       f"epoch_eta:{eta_epoch}, time_all:{t:.3f}, time_data:{t_d:.3f}, lr:{lr:.3e}, s:({data_info['resolution'][0][0].item()}, {data_info['resolution'][0][1].item()}), "
                info += ', '.join([f"{k}:{v:.4f}" for k, v in log_buffer.output.items()])
                logger.info(info)
                last_tic = time.time()
                log_buffer.clear()
                data_time_all = 0
            logs.update(lr=lr)
            accelerator.log(logs, step=global_step + start_step)

            global_step += 1
            data_time_start= time.time()

            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                if ((epoch - 1) * len(train_dataloader) + step + 1) % config.save_model_steps == 0:
                    save_path = os.path.join(os.path.join(config.work_dir, 'checkpoints'), f"checkpoint-{(epoch - 1) * len(train_dataloader) + step + 1}")
                    os.umask(0o000)
                    logger.info(f"Start to save state to {save_path}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")


        accelerator.wait_for_everyone()
        if epoch % config.save_model_epochs == 0 or epoch == config.num_epochs:
            os.umask(0o000)
            save_path = os.path.join(os.path.join(config.work_dir, 'checkpoints'), f"checkpoint-{(epoch - 1) * len(train_dataloader) + step + 1}")
            logger.info(f"Start to save state to {save_path}")
            model = accelerator.unwrap_model(model)
            model.save_pretrained(save_path)
            lora_state_dict = get_peft_model_state_dict(model, adapter_name="default")
            StableDiffusionPipeline.save_lora_weights(os.path.join(save_path, "transformer_lora"), lora_state_dict)
            logger.info(f"Saved state to {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("config", type=str, help="config")
    parser.add_argument("--cloud", action='store_true', default=False, help="cloud or local machine")
    parser.add_argument("--work-dir", default='output', help='the dir to save logs and models')
    parser.add_argument("--resume-from", help='the dir to save logs and models')
    parser.add_argument("--local-rank", type=int, default=-1)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--lora_rank", type=int, default=64, help="The rank of the LoRA projection matrix.", )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    config = read_config(args.config)

    config.resume_from = None
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        config.work_dir = args.work_dir
    if args.cloud:
        config.data_root = '/data/data'
    if args.resume_from is not None:
        config.resume_from = args.resume_from
    if args.debug:
        config.log_interval = 1
        config.train_batch_size = 4
        config.valid_num = 10
        config.save_model_steps = 10

    os.umask(0o000)
    os.makedirs(config.work_dir, exist_ok=True)

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

    logger.info(accelerator.state)
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

    # prepare null_embedding for training
    if not os.path.exists('output/pretrained_models/null_embed.pth'):
        logger.info(f"Creating output/pretrained_models/null_embed.pth")
        os.makedirs('output/pretrained_models/', exist_ok=True)
        pipe = PixArtAlphaPipeline.from_pretrained("PixArt-alpha/PixArt-XL-2-1024-MS", torch_dtype=torch.float16, use_safetensors=True,).to("cuda")
        torch.save(pipe.encode_prompt(""), 'output/pretrained_models/null_embed.pth')
        del pipe
        torch.cuda.empty_cache()

    # build models
    train_diffusion = IDDPM(str(config.train_sampling_steps), learn_sigma=learn_sigma, pred_sigma=pred_sigma, return_startx=True)
    model_teacher = Transformer2DModel.from_pretrained(config.load_from, subfolder="transformer")
    model_teacher.requires_grad_(False)
    model = Transformer2DModel.from_pretrained(config.load_from, subfolder="transformer").train()
    logger.info(f"{model.__class__.__name__} Model Parameters: {sum(p.numel() for p in model.parameters()):}")

    lora_config = LoraConfig(
        r=config.lora_rank,
        target_modules=[
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
            "proj_in",
            "proj_out",
            "ff.net.0.proj",
            "ff.net.2",
            "proj",
            "linear",
            "linear_1",
            "linear_2",
            # "scale_shift_table",      # not available due to the implementation in huggingface/peft, working on it.
        ],
    )
    print(lora_config)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 9. Handle mixed precision and device placement
    # For mixed precision training we cast all non-trainable weigths to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # 11. Enable optimizations
    # model.enable_xformers_memory_efficient_attention()
    # model_teacher.enable_xformers_memory_efficient_attention()

    lora_layers = filter(lambda p: p.requires_grad, model.parameters())

    # for name, params in model.named_parameters():
    #     if params.requires_grad == False: logger.info(f"freeze param: {name}")
    #
    # for name, params in model.named_parameters():
    #     if params.requires_grad == True: logger.info(f"trainable param: {name}")

    # 10. Handle saving and loading of checkpoints
    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                transformer_ = accelerator.unwrap_model(models[0])
                lora_state_dict = get_peft_model_state_dict(transformer_, adapter_name="default")
                StableDiffusionPipeline.save_lora_weights(os.path.join(output_dir, "transformer_lora"), lora_state_dict)
                # save weights in peft format to be able to load them back
                transformer_.save_pretrained(output_dir)

                for _, model in enumerate(models):
                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            # load the LoRA into the model
            transformer_ = accelerator.unwrap_model(models[0])
            transformer_.load_adapter(input_dir, "default", is_trainable=True)

            for _ in range(len(models)):
                # pop models so that they are not loaded again
                models.pop()

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if config.grad_checkpointing:
        model.enable_gradient_checkpointing()

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
        train_dataloader = build_dataloader(dataset, num_workers=config.num_workers, batch_size=config.train_batch_size, shuffle=True)

    # build optimizer and lr scheduler
    lr_scale_ratio = 1
    if config.get('auto_lr', None):
        lr_scale_ratio = auto_scale_lr(config.train_batch_size * get_world_size() * config.gradient_accumulation_steps,
                                       config.optimizer,
                                       **config.auto_lr)
    optimizer = build_optimizer(model, config.optimizer)
    lr_scheduler = build_lr_scheduler(config, optimizer, train_dataloader, lr_scale_ratio)

    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())

    if accelerator.is_main_process:
        accelerator.init_trackers(f"tb_{timestamp}")

    start_epoch = 0
    start_step = 0
    total_steps = len(train_dataloader) * config.num_epochs

    solver = DDIMSolver(train_diffusion.alphas_cumprod, timesteps=config.train_sampling_steps, ddim_timesteps=config.num_ddim_timesteps)
    solver.to(accelerator.device)

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, model_teacher = accelerator.prepare(model, model_teacher)
    optimizer, train_dataloader, lr_scheduler = accelerator.prepare(optimizer, train_dataloader, lr_scheduler)

    if config.resume_from is not None:
        if config.resume_from != "latest":
            path = os.path.basename(config.resume_from)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(os.path.join(config.work_dir, 'checkpoints'))
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(f"Checkpoint '{config.resume_from}' does not exist. Starting a new training run.")
            config.resume_from = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(config.work_dir, 'checkpoints', path))
            start_step = int(path.split("-")[1])
            start_epoch = start_step // len(train_dataloader)

    train(model)