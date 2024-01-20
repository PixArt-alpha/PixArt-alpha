import argparse
import os
from datetime import datetime
import numpy as np
import sys
from pathlib import Path
from typing import List, Union

current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))

import gradio as gr
from gradio.components import Textbox, Image, Slider
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.utils import _log_api_usage_once, make_grid, save_image

from diffusion import IDDPM, DPMS, SASolverSampler
from diffusion.data.datasets import *
from diffusion.model.hed import HEDdetector
from diffusion.model.nets import PixArtMS_XL_2, ControlPixArtHalf, ControlPixArtMSHalf
from diffusion.model.t5 import T5Embedder
from diffusion.model.utils import prepare_prompt_ar, resize_and_crop_tensor
from diffusion.utils.misc import read_config
from diffusers.models import AutoencoderKL
from tools.download import find_model

vae_scale = 0.18215

DESCRIPTION = """![Logo](https://raw.githubusercontent.com/PixArt-alpha/PixArt-alpha.github.io/master/static/images/logo.png)
        # PixArt-Alpha 1024px + ControlNet. This is the demo for ControlNet combined with 1024px PixArt-Alpha.
        # The input reference image need to be around 1024x1024. And descriptive prompts also need to be provided.
        # You may change the random seed, if you didn't get satisfied results.
        """


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="config")
    parser.add_argument('--num_sampling_steps', default=14, type=int)
    parser.add_argument('--cfg_scale', default=4.5, type=int)
    parser.add_argument('--image_size', default=1024, type=int)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--tokenizer_path', default='output/pretrained_models/sd-vae-ft-ema', type=str)

    parser.add_argument('--llm_model', default='t5', type=str)

    parser.add_argument('--sampling_algo', default='dpm-solver', type=str, choices=['iddpm', 'dpm-solver', 'sa-solver'])

    parser.add_argument('--port', default=7788, type=int)
    parser.add_argument('--condition_strength', default=1, type=float)

    return parser.parse_args()


@torch.no_grad()
def ndarr_image(tensor: Union[torch.Tensor, List[torch.Tensor]], **kwargs, ) -> None:
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(save_image)
    grid = make_grid(tensor, **kwargs)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    return ndarr


def set_env():
    torch.manual_seed(0)
    torch.set_grad_enabled(False)


@torch.inference_mode()
def generate_img(prompt, given_image, seed):
    torch.manual_seed(seed)
    torch.cuda.empty_cache()
    strength = 1.0
    c_vis = given_image

    save_promt_path = f'{save_prompt_path}/tested_prompts{datetime.now().date()}.txt'
    with open(save_promt_path, 'a') as f:
        f.write(prompt + '\n')
    prompt_clean, prompt_show, hw, ar, custom_hw = prepare_prompt_ar(prompt, base_ratios, device=device)  # ar for aspect ratio
    prompt_clean = prompt_clean.strip()
    if isinstance(prompt_clean, str):
        prompts = [prompt_clean]

    caption_embs, emb_masks = llm_embed_model.get_text_embeddings(prompts)
    caption_embs = caption_embs[:, None]

    null_y = model.y_embedder.y_embedding[None].repeat(len(prompts), 1, 1)[:, None]

    # condition process
    if given_image is not None:
        ar = torch.tensor([given_image.size[1] / given_image.size[0]], device=device)[None]
        custom_hw = torch.tensor([given_image.size[1], given_image.size[0]], device=device)[None]
        closest_hw = base_ratios[min(base_ratios.keys(), key=lambda ratio: abs(float(ratio) - ar))]
        hw = torch.tensor(closest_hw, device=device)[None]
        condition_transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB')),
            T.Resize(int(min(closest_hw))),
            T.CenterCrop([int(closest_hw[0]), int(closest_hw[1])]),
            T.ToTensor(),
        ])

        given_image = condition_transform(given_image).unsqueeze(0).to(device)
        hed_edge = hed(given_image) * strength
        hed_edge = TF.normalize(hed_edge, [.5], [.5])
        hed_edge = hed_edge.repeat(1, 3, 1, 1)
        posterior = vae.encode(hed_edge).latent_dist
        condition = posterior.sample()
        c = condition * vae_scale
        c_vis = vae.decode(condition)['sample']
        c_vis = torch.clamp(127.5 * c_vis + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()[0]
    else:
        c = None

    latent_size_h, latent_size_w = int(hw[0, 0] // 8), int(hw[0, 1] // 8)
    # Sample images:
    if args.sampling_algo == 'iddpm':
        # Create sampling noise:
        n = len(prompts)
        z = torch.randn(n, 4, latent_size, latent_size, device=device).repeat(2, 1, 1, 1)
        model_kwargs = dict(y=torch.cat([caption_embs, null_y]), cfg_scale=args.cfg_scale,
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

    elif args.sampling_algo == 'sa-solver':
        # Create sampling noise:
        n = len(prompts)
        model_kwargs = dict(data_info={'img_hw': hw, 'aspect_ratio': ar}, mask=emb_masks, c=c)
        sas_solver = SASolverSampler(model.forward_with_dpmsolver, device=device)
        samples = sas_solver.sample(
            S=args.num_sampling_steps,
            batch_size=n,
            shape=(4, latent_size_h, latent_size_w),
            eta=1,
            conditioning=caption_embs,
            unconditional_conditioning=null_y,
            unconditional_guidance_scale=args.cfg_scale,
            model_kwargs=model_kwargs,
        )[0]

    samples = vae.decode(samples / vae_scale).sample
    torch.cuda.empty_cache()
    samples = resize_and_crop_tensor(samples, custom_hw[0, 1], custom_hw[0, 0])

    return ndarr_image(samples, normalize=True, value_range=(-1, 1)), c_vis, prompt_show


if __name__ == '__main__':
    args = get_args()
    config = read_config(args.config)
    set_env()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_prompt_path = 'output/demo/online_demo_prompts/'
    os.makedirs(save_prompt_path, exist_ok=True)

    assert args.image_size in [512, 1024], "We only provide pre-trained models for 512x512 and 1024x1024 resolutions."
    lewei_scale = {512: 1, 1024: 2}
    latent_size = args.image_size // 8
    weight_dtype = torch.float16
    print(f"Inference with {weight_dtype}")

    model = PixArtMS_XL_2(input_size=latent_size, lewei_scale=lewei_scale[args.image_size])
    if config.image_size == 512:
        print('model architecture ControlPixArtHalf and image size is 512')
        model = ControlPixArtHalf(model).to(device)
    elif config.image_size == 1024:
        print('model architecture ControlPixArtMSHalf and image size is 1024')
        model = ControlPixArtMSHalf(model).to(device)

    state_dict = find_model(args.model_path)['state_dict']
    if 'pos_embed' in state_dict:
        del state_dict['pos_embed']
    elif 'base_model.pos_embed' in state_dict:
        del state_dict['base_model.pos_embed']
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print('Missing keys (missing pos_embed is normal): ', missing)
    print('Unexpected keys', unexpected)
    model.eval()
    model.to(weight_dtype)
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

    gr.Markdown(DESCRIPTION)
    demo = gr.Interface(fn=generate_img,
                        inputs=[
                            Textbox(label="Enter a reference image, the resolution of image need around 1024 x 1024",
                                    placeholder="Please enter your prompt. \n"),
                            Image(type="pil", label="Condition"),
                            Slider(minimum=0., maximum=10000., value=0, step=2, label='seed'),
                            ],
                        outputs=[Image(type="numpy", label="Img"),
                                 Image(type="numpy", label="HED Edge Map"),
                                 Textbox(label="clean prompt"),]
                        )
    demo.queue(max_size=20).launch(server_name="0.0.0.0", server_port=args.port, debug=True)
