import argparse
import sys
from pathlib import Path
current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))
import torch
from torchvision.utils import save_image
from diffusion import IDDPM, DPMS, SASolverSampler
from diffusers.models import AutoencoderKL
from tools.download import find_model
from datetime import datetime
from typing import List, Union
import gradio as gr
from gradio.components import Textbox, Image, Slider
from diffusion.model.utils import prepare_prompt_ar, resize_and_crop_tensor
from diffusion.model.nets import PixArtMS_XL_2, PixArtMS
from diffusion.model.nets import ControlT2IDiT, ControlPixArt_Mid, ControlPixArtAll, ControlPixArtHalf
from diffusion.model.t5 import T5Embedder
from torchvision.utils import _log_api_usage_once, make_grid
from diffusion.data.datasets import *
from diffusion.model.hed import HEDdetector
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import ImageFilter
from PIL import Image as Image_PIL
from diffusion.utils.misc import set_random_seed, read_config, init_random_seed, DebugUnderflowOverflow #MoxingWorker
from diffusion.data.builder import build_dataset, build_dataloader, set_data_root
import os
import numpy as np 
# The code of testing on the train set or test set

vae_scale = 0.18215

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="config")
    parser.add_argument('--num_sampling_steps', default=14, type=int)
    parser.add_argument('--cfg_scale', default=4.5, type=int)
    parser.add_argument('--image_size', default=1024, type=int)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--exp_id', type=str)
    parser.add_argument('--step', type=int)
    parser.add_argument('--tokenizer_path', default='output/pretrained_models/sd-vae-ft-ema', type=str)

    parser.add_argument('--txt_feature_root', default='data/SA1B/caption_feature', type=str)
    parser.add_argument('--txt_feature_path', default='data/SA1B/partition/part0.txt', type=str)

    parser.add_argument('--llm_model', default='t5', type=str)
    parser.add_argument('--test_mode', default='inference', type=str, help='choose from train or inference mode')

    parser.add_argument('--sampling_algo', default='dpm-solver', type=str, choices=['iddpm', 'dpm-solver', 'sa-solver'])

    parser.add_argument('--port', default=7788, type=int)
    parser.add_argument('--controlnet_type', default='all', type=str)
    parser.add_argument('--start_index', default=0, type=int)

    return parser.parse_args()


@torch.no_grad()
def ndarr_image(tensor: Union[torch.Tensor, List[torch.Tensor]], **kwargs,) -> None:
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(save_image)
    grid = make_grid(tensor, **kwargs)
    # Add 0.5 after unnormalizing to [0, 255] to round to the nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    return ndarr


def set_env():
    torch.manual_seed(0)
    torch.set_grad_enabled(False)

def mask_feature(emb, mask):
    if emb.shape[0] == 1:
        keep_index = mask.sum().item()
        # masked_feature= emb * mask.unsqueeze(dim=-1)
        return emb[:, :, :keep_index, :], keep_index
    else:
        masked_feature= emb * mask[:, None, :, None]
        return masked_feature, emb.shape[2]


def prepare_input_test(prompt, condition, strength, radius):
    torch.cuda.empty_cache()
    if condition is not None:
        condition = condition_transform(condition).unsqueeze(0).to(device)
        condition = hed(condition) * strength
        if radius > 0:
            condition[0] = TF.gaussian_blur(condition[0], kernel_size=radius)
        TF.to_pil_image(condition[0]).save('./test.png')
        # condition = TF.to_tensor(condition).unsqueeze(0).to(device)
        condition = TF.normalize(condition, [.5], [.5])
        condition = condition.repeat(1, 3, 1, 1) 
        posterior = vae.encode(condition).latent_dist
        c = posterior.sample()
    else:
        c = None
        
    if c is not None:
        c = c * vae_scale

    save_promt_path = f'output_demo/demo/online_demo_prompts/tested_prompts{datetime.now().date()}.txt'
    with open(save_promt_path, 'a') as f:
        f.write(prompt + '\n')

    return c, prompt

def prepare_input_train(data_info):
    # prepare condition
    prompt = data_info['prompt']
    condition = data_info['condition'].to(device)
    # TF.to_pil_image(condition[0]).save('./test.png')
    c_vis = vae.decode(condition)['sample']
    c = condition * vae_scale
    
    prompt = data_info['prompt'][0].split("/")[-1]
    
    return c, prompt, c_vis



@torch.inference_mode()
def sample_image(c, prompt):
    prompt_clean, prompt_show, hw, ar, custom_hw = prepare_prompt_ar(prompt, base_ratios, device=device)      # ar for aspect ratio
    prompt_clean = prompt_clean.strip()
    if isinstance(prompt_clean, str):
        prompts = [prompt_clean]

    caption_embs, emb_masks = llm_embed_model.get_text_embeddings(prompts)
    caption_embs = caption_embs[:, None]

    null_y = model.y_embedder.y_embedding[None].repeat(len(prompts), 1, 1)[:, None]


    latent_size_h, latent_size_w = int(hw[0, 0]//8), int(hw[0, 1]//8)
    # Sample images:
    if args.sampling_algo == 'iddpm':
        # Create sampling noise:
        n = len(prompts)
        z = torch.randn(n, 4, latent_size, latent_size, device=device).repeat(2, 1, 1, 1)
        model_kwargs = dict(y=torch.cat([caption_embs, null_y]),
                            cfg_scale=args.cfg_scale,
                            data_info={'img_hw': hw, 'aspect_ratio': ar},
                            mask=emb_masks, c=c)
        diffusion = IDDPM(str(args.num_sampling_steps))
        samples = diffusion.p_sample_loop(
            model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True,
            device=device
        )
        samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    elif args.sampling_algo == 'dpm-solver':
        # Create sampling noise:
        n = len(prompts)
        z = torch.randn(n, 4, latent_size_h, latent_size_w, device=device)
        model_kwargs = dict(data_info={'img_hw': hw, 'aspect_ratio': ar}, mask=emb_masks, c=c)
        dpm_solver = DPMS(model.forward_with_dpmsolver,
                          condition=caption_embs,
                          uncondition=null_y,
                          cfg_scale=args.cfg_scale,
                          model_kwargs=model_kwargs)
        samples = dpm_solver.sample(
            z,
            steps=args.num_sampling_steps,
            order=2,
            skip_type="time_uniform",
            method="multistep",
        )
        samples = vae.decode(samples / vae_scale).sample
        torch.cuda.empty_cache()
        samples = resize_and_crop_tensor(samples, custom_hw[0,1], custom_hw[0,0])

        # unconditional output
        model_kwargs = dict(data_info={'img_hw': hw, 'aspect_ratio': ar}, mask=emb_masks, c=None)
        dpm_solver = DPMS(model.forward_with_dpmsolver,
                          condition=caption_embs,
                          uncondition=null_y,
                          cfg_scale=args.cfg_scale,
                          model_kwargs=model_kwargs)
        samples_uncon = dpm_solver.sample(
            z,
            steps=args.num_sampling_steps,
            order=2,
            skip_type="time_uniform",
            method="multistep",
        )
        samples_uncon = vae.decode(samples_uncon / vae_scale).sample
        torch.cuda.empty_cache()
        samples_uncon = resize_and_crop_tensor(samples_uncon, custom_hw[0,1], custom_hw[0,0])

    elif args.sampling_algo == 'sa-solver':
        # Create sampling noise:
        n = len(prompts)
        model_kwargs = dict(data_info={'img_hw': hw, 'aspect_ratio': ar}, mask=emb_masks, c=c)
        sas_solver = SASolverSampler(model.forward_with_dpmsolver, device=device)
        samples = sas_solver.sample(
            S=20,
            batch_size=n,
            shape=(4, latent_size_h, latent_size_w),
            eta=1,
            conditioning=caption_embs,
            unconditional_conditioning=null_y,
            unconditional_guidance_scale=args.cfg_scale,
            model_kwargs=model_kwargs,
        )[0]
    
    display_model_info = f'Model path: {args.model_path},\nBase image size: {args.image_size}, \nSampling Algo: {args.sampling_algo}'
    return ndarr_image(samples, normalize=True, value_range=(-1, 1)), ndarr_image(samples_uncon, normalize=True, value_range=(-1, 1)), prompt_show, display_model_info


if __name__ == '__main__':
    args = get_args()
    config = read_config(args.config)
    set_env()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    assert args.image_size in [256, 512, 1024], "We only provide pre-trained models for 256x256, 512x512 and 1024x1024 resolutions."
    lewei_scale = {256: 1, 512: 1, 1024: 2}
    latent_size = args.image_size // 8
    
    if args.controlnet_type == 'all':
        model = PixArtMS_XL_2(input_size=latent_size, lewei_scale=lewei_scale[args.image_size])
        model = ControlPixArtAll(model).to(device)
    else:
        model = PixArtMS(input_size=latent_size, lewei_scale=lewei_scale[args.image_size])
        model = ControlPixArtHalf(model).to(device)

    state_dict = find_model(args.model_path)['state_dict']
    if 'pos_embed' in state_dict:
        del state_dict['pos_embed']
    elif 'base_model.pos_embed' in state_dict:
        del state_dict['base_model.pos_embed']
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print('Missing keys: ', missing)
    print('Unexpected keys', unexpected)
    model.eval()
    display_model_info = f'model path: {args.model_path},\n base image size: {args.image_size}'
    base_ratios = eval(f'ASPECT_RATIO_{args.image_size}_TEST')

    vae = AutoencoderKL.from_pretrained(args.tokenizer_path).to(device)
    hed = HEDdetector(False).to(device)

    if args.llm_model == 't5':
        print("begin load t5")
        llm_embed_model = T5Embedder(device=device, local_cache=True, cache_dir='data/t5_ckpts', torch_dtype=torch.float)
        print("finish load t5")
    else:
        print(f'We support t5 only, please initialize the llm again')
        sys.exit()
    condition_transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB')),
        T.Resize(args.image_size),  # Image.BICUBIC
        T.CenterCrop(args.image_size),
        T.ToTensor(),
    ])
    
    # the mode is train | inference | demo
    if args.test_mode in ['train', 'inference']:
        # loading training dataset
        set_data_root(config.data_root)
        dataset = build_dataset(config.data, resolution=config.image_size, aspect_ratio_type=config.aspect_ratio_type, train_ratio=0.001, mode=args.test_mode)
        train_dataloader = build_dataloader(dataset, num_workers=1, batch_size=1, shuffle=False)
        cnt = 0
        output_folder = f'output_demo/demo/exp_{args.exp_id}_mode_{args.test_mode}/'
        os.makedirs(output_folder, exist_ok=True)
        output_folder = f'{output_folder}/step_{args.step}/'
        os.makedirs(output_folder, exist_ok=True)
        save_promt_path = f'{output_folder}/prompt.txt'
        save_image_path = f'{output_folder}/images/'
        os.makedirs(save_image_path, exist_ok=True)
        
        for index, batch in enumerate(train_dataloader):
            # try:
            # if cnt > 80:
            #     break
            if index < args.start_index - 1:
                continue
            data_info = batch[3]
            c, prompt, c_vis = prepare_input_train(data_info)
            image, image_uncon, prompt_show, display_info = sample_image(c, prompt)
            with open(save_promt_path, 'a') as f:
                f.write(f"number_{index}_{prompt}\n")
            c_vis = c_vis.permute(0, 2, 3, 1)[0].cpu()
            c_vis = (c_vis*255).clip(0, 255).int().numpy()
            vis = np.concatenate([c_vis, image, image_uncon], axis=1)
            Image_PIL.fromarray(vis.astype(np.uint8)).save(f'{save_image_path}/{index:04d}.png')
            cnt = cnt + 1
            # except:
            #     continue
    # else:
    #     demo = gr.Interface(fn=generate_img,
    #                         inputs=[Textbox(label="Begin your magic",
    #                                     placeholder="Please enter your prompt. \n"
    #                                                 "If you want to specify a aspect ratio or determine a customized height and width, "
    #                                                 "use --ar h:w (or --aspect_ratio h:w) or --hw h:w. If no aspect ratio or hw is given, all setting will be default."),
    #                                 Image(type="pil", label="Condition"),
    #                                 Slider(minimum=0., maximum=1., value=1., label='edge strength'),
    #                                 Slider(minimum=-1., maximum=99., value=-1, step=2, label='radius'),
    #                                 ],
    #                         outputs=[Image(type="numpy", label="Img"),
    #                                 Textbox(label="clean prompt"),
    #                                 Textbox(label="model info")],)
    #     demo.launch(server_name="0.0.0.0", server_port=args.port, debug=True, share=True)

