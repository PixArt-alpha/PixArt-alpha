import argparse
import sys
from pathlib import Path
current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))
import os
import torch
from torchvision.utils import save_image
from diffusion import IDDPM, DPMS
from diffusers.models import AutoencoderKL
from download import find_model
from datetime import datetime
from typing import List, Union
import gradio as gr
from gradio.components import Textbox, Image
from diffusion.model.utils import prepare_prompt_ar, resize_img
from diffusion.model.nets.PixelArtMS import PixelArtMS_XL_2
from diffusion.model.t5 import T5Embedder
from torchvision.utils import _log_api_usage_once, make_grid
from diffusion.data.datasets import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_sampling_steps', default=100, type=int)
    parser.add_argument('--cfg_scale', default=4.5, type=int)
    parser.add_argument('--image_size', default=256, type=int)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--tokenizer_path', default='output/pretrained_models/sd-vae-ft-ema', type=str)

    parser.add_argument('--txt_feature_root', default='data/SA1B/caption_feature', type=str)
    parser.add_argument('--txt_feature_path', default='data/SA1B/partition/part0.txt', type=str)

    parser.add_argument('--llm_model', default='t5', type=str)

    parser.add_argument('--sampling_algo', default='iddpm', type=str, choices=['iddpm', 'dpms'])

    parser.add_argument('--port', default=7788, type=int)

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

@torch.inference_mode()
def generate_img(prompt):
    torch.cuda.empty_cache()
    os.makedirs(f'output/demo/online_demo_prompts/', exist_ok=True)
    save_promt_path = f'output/demo/online_demo_prompts/tested_prompts{datetime.now().date()}.txt'
    with open(save_promt_path, 'a') as f:
        f.write(prompt + '\n')
    print(prompt)
    prompt_clean, prompt_show, hw, ar, custom_hw = prepare_prompt_ar(prompt, base_ratios, device=device)      # ar for aspect ratio
    prompt_clean = prompt_clean.strip()
    if isinstance(prompt_clean, str):
        prompts = [prompt_clean]

    caption_embs, emb_masks = llm_embed_model.get_text_embeddings(prompts)
    caption_embs = caption_embs[:, None]
    masked_embs, keep_index = mask_feature(caption_embs, emb_masks)
    masked_embs = masked_embs.cuda()

    null_y = model.y_embedder.y_embedding[None].repeat(len(prompts), 1, 1)[:, None]

    latent_size_h, latent_size_w = int(hw[0, 0]//8), int(hw[0, 1]//8)
    # Sample images:
    if args.sampling_algo == 'iddpm':
        # Create sampling noise:
        n = len(prompts)
        z = torch.randn(n, 4, latent_size_h, latent_size_w, device=device).repeat(2, 1, 1, 1)
        # model_kwargs = dict(y=torch.cat([masked_embs, null_y], dim=0), cfg_scale=cfg_scale)
        model_kwargs = dict(y=torch.cat([masked_embs, null_y[:, :, :keep_index, :]]),
                            cfg_scale=args.cfg_scale,
                            data_info={'img_hw': hw, 'aspect_ratio': ar})
        diffusion = IDDPM(str(args.num_sampling_steps))
        samples = diffusion.p_sample_loop(
            model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True,
            device=device
        )
        samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    elif args.sampling_algo == 'dpms':
        # Create sampling noise:
        n = len(prompts)
        z = torch.randn(n, 4, latent_size_h, latent_size_w, device=device)
        model_kwargs = dict(data_info={'img_hw': hw, 'aspect_ratio': ar})
        dpm_solver = DPMS(model.forward_with_dpmsolver,
                          condition=masked_embs,
                          uncondition=null_y[:, :, :keep_index, :],
                          cfg_scale=args.cfg_scale,
                          model_kwargs=model_kwargs)
        samples = dpm_solver.sample(
            z,
            steps=20,
            order=2,
            skip_type="time_uniform",
            method="multistep",
        )
    samples = vae.decode(samples / 0.18215).sample
    torch.cuda.empty_cache()
    samples = resize_img(samples, hw, custom_hw)
    return ndarr_image(samples, normalize=True, value_range=(-1, 1)), prompt_show, display_model_info

if __name__ == '__main__':
    args = get_args()
    set_env()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    assert args.image_size in [256, 512, 1024], "We only provide pre-trained models for 256x256, 512x512 and 1024x1024 resolutions."
    lewei_scale = {256: 1, 512: 1, 1024: 2}
    latent_size = args.image_size // 8
    t5_device = {256: 'cuda', 512: 'cuda', 1024: 'cuda'}
    model = PixelArtMS_XL_2(input_size=latent_size, lewei_scale=lewei_scale[args.image_size]).to(device)
    state_dict = find_model(args.model_path)
    del state_dict['state_dict']['pos_embed']
    missing, unexpected = model.load_state_dict(state_dict['state_dict'], strict=False)
    print('Missing keys: ', missing)
    print('Unexpected keys', unexpected)
    model.eval()
    display_model_info = f'Model path: {args.model_path},\n Base image size: {args.image_size}, \n Sampling Algo: {args.sampling_algo}'
    base_ratios = eval(f'ASPECT_RATIO_{args.image_size}_TEST')

    vae = AutoencoderKL.from_pretrained(args.tokenizer_path).to(device)

    if args.llm_model == 't5':
        llm_embed_model = T5Embedder(device=t5_device[args.image_size], local_cache=True, cache_dir='data/t5_ckpts', torch_dtype=torch.float)
    else:
        print(f'We support t5 only, please initialize the llm again')
        sys.exit()

    # prompt = 'dog'
    # generate_img(prompt)

    demo = gr.Interface(fn=generate_img,
                        inputs=Textbox(label="Begin your magic "
                                             "(If you want to specify a aspect ratio or determine a customized height and width, "
                                             "use --ar h:w (or --aspect_ratio h:w) or --hw h:w. If no aspect ratio or hw is given, all setting will be default.)",
                                       placeholder="Please enter your prompt. \n"),
                        outputs=[Image(type="numpy", label="Img"),
                                 Textbox(label="clean prompt"),
                                 Textbox(label="model info")],)
    demo.launch(server_name="0.0.0.0", server_port=args.port, debug=True, enable_queue=True)