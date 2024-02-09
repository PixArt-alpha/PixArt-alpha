import argparse
import sys
from pathlib import Path
current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))
import os
import random
import torch
from torchvision.utils import save_image
from diffusion import IDDPM, DPMS, SASolverSampler
from diffusers.models import AutoencoderKL
from tools.download import find_model
from datetime import datetime
from typing import List, Union
import gradio as gr
import numpy as np
from gradio.components import Textbox, Image
from diffusion.model.utils import prepare_prompt_ar, resize_and_crop_tensor
from diffusion.model.nets import PixArtMS_XL_2, PixArt_XL_2
from diffusion.model.t5 import T5Embedder
from torchvision.utils import _log_api_usage_once, make_grid
from diffusion.data.datasets import ASPECT_RATIO_512_TEST, ASPECT_RATIO_1024_TEST
from asset.examples import examples


MAX_SEED = np.iinfo(np.int32).max


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', default=1024, type=int)
    parser.add_argument('--model_path', default='output/pretrained_models/PixArt-XL-2-1024-MS.pth', type=str)
    parser.add_argument('--t5_path', default='output/pretrained_models', type=str)
    parser.add_argument('--tokenizer_path', default='output/pretrained_models/sd-vae-ft-ema', type=str)
    parser.add_argument('--llm_model', default='t5', type=str)
    parser.add_argument('--port', default=7788, type=int)

    return parser.parse_args()


@torch.no_grad()
def ndarr_image(tensor: Union[torch.Tensor, List[torch.Tensor]], **kwargs,) -> None:
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(save_image)
    grid = make_grid(tensor, **kwargs)
    # Add 0.5 after unnormalizing to [0, 255] to round to the nearest integer
    return grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()


def set_env(seed=0):
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)
    for _ in range(30):
        torch.randn(1, 4, args.image_size, args.image_size)


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


@torch.inference_mode()
def generate_img(prompt, sampler, sample_steps, scale, seed=0, randomize_seed=False):
    seed = int(randomize_seed_fn(seed, randomize_seed))
    set_env(seed)

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

    null_y = model.y_embedder.y_embedding[None].repeat(len(prompts), 1, 1)[:, None]

    latent_size_h, latent_size_w = int(hw[0, 0]//8), int(hw[0, 1]//8)
    # Sample images:
    if sampler == 'iddpm':
        # Create sampling noise:
        n = len(prompts)
        z = torch.randn(n, 4, latent_size_h, latent_size_w, device=device).repeat(2, 1, 1, 1)
        model_kwargs = dict(y=torch.cat([caption_embs, null_y]),
                            cfg_scale=scale, data_info={'img_hw': hw, 'aspect_ratio': ar}, mask=emb_masks)
        diffusion = IDDPM(str(sample_steps))
        samples = diffusion.p_sample_loop(
            model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True,
            device=device
        )
        samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    elif sampler == 'dpm-solver':
        # Create sampling noise:
        n = len(prompts)
        z = torch.randn(n, 4, latent_size_h, latent_size_w, device=device)
        model_kwargs = dict(data_info={'img_hw': hw, 'aspect_ratio': ar}, mask=emb_masks)
        dpm_solver = DPMS(model.forward_with_dpmsolver,
                          condition=caption_embs,
                          uncondition=null_y,
                          cfg_scale=scale,
                          model_kwargs=model_kwargs)
        samples = dpm_solver.sample(
            z,
            steps=sample_steps,
            order=2,
            skip_type="time_uniform",
            method="multistep",
        )
    elif sampler == 'sa-solver':
        # Create sampling noise:
        n = len(prompts)
        model_kwargs = dict(data_info={'img_hw': hw, 'aspect_ratio': ar}, mask=emb_masks)
        sa_solver = SASolverSampler(model.forward_with_dpmsolver, device=device)
        samples = sa_solver.sample(
            S=sample_steps,
            batch_size=n,
            shape=(4, latent_size_h, latent_size_w),
            eta=1,
            conditioning=caption_embs,
            unconditional_conditioning=null_y,
            unconditional_guidance_scale=scale,
            model_kwargs=model_kwargs,
        )[0]
    samples = vae.decode(samples / 0.18215).sample
    torch.cuda.empty_cache()
    samples = resize_and_crop_tensor(samples, custom_hw[0,1], custom_hw[0,0])
    display_model_info = f'Model path: {args.model_path},\nBase image size: {args.image_size}, \nSampling Algo: {sampler}'
    return ndarr_image(samples, normalize=True, value_range=(-1, 1)), prompt_show, display_model_info, seed


if __name__ == '__main__':
    from diffusion.utils.logger import get_root_logger
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = get_root_logger()

    assert args.image_size in [512, 1024], "We only provide pre-trained models for 256x256, 512x512 and 1024x1024 resolutions."
    lewei_scale = {512: 1, 1024: 2}
    latent_size = args.image_size // 8
    t5_device = {512: 'cuda', 1024: 'cuda'}
    if args.image_size == 512:
        model = PixArt_XL_2(input_size=latent_size, lewei_scale=lewei_scale[args.image_size]).to(device)
    else:
        model = PixArtMS_XL_2(input_size=latent_size, lewei_scale=lewei_scale[args.image_size]).to(device)
    state_dict = find_model(args.model_path)
    del state_dict['state_dict']['pos_embed']
    missing, unexpected = model.load_state_dict(state_dict['state_dict'], strict=False)
    logger.warning(f'Missing keys: {missing}')
    logger.warning(f'Unexpected keys: {unexpected}')
    model.eval()
    base_ratios = eval(f'ASPECT_RATIO_{args.image_size}_TEST')

    vae = AutoencoderKL.from_pretrained(args.tokenizer_path).to(device)

    if args.llm_model == 't5':
        llm_embed_model = T5Embedder(device=t5_device[args.image_size], local_cache=True, cache_dir=args.t5_path, torch_dtype=torch.float)
    else:
        print('We support t5 only, please initialize the llm again')
        sys.exit()

    title = f"""
        '' Unleashing your Creativity \n ''
        <div style='display: flex; align-items: center; justify-content: center; text-align: center;'>
            <img src='https://raw.githubusercontent.com/PixArt-alpha/PixArt-alpha.github.io/master/static/images/logo.png' style='width: 400px; height: auto; margin-right: 10px;' />
            {args.image_size}px
        </div>
    """
    DESCRIPTION = """# PixArt-Alpha 1024px
            ## If PixArt-Alpha is helpful, please help to ⭐ the [Github Repo](https://github.com/PixArt-alpha/PixArt) and recommend it to your friends 😊'
            #### [PixArt-Alpha 1024px](https://github.com/PixArt-alpha/PixArt-alpha) is a transformer-based text-to-image diffusion system trained on text embeddings from T5. This demo uses the [PixArt-alpha/PixArt-XL-2-1024-MS](https://huggingface.co/PixArt-alpha/PixArt-XL-2-1024-MS) checkpoint.
            #### English prompts ONLY; 提示词仅限英文
            Don't want to queue? Try [OpenXLab](https://openxlab.org.cn/apps/detail/PixArt-alpha/PixArt-alpha) or [Google Colab Demo](https://colab.research.google.com/drive/1jZ5UZXk7tcpTfVwnX33dDuefNMcnW9ME?usp=sharing).
            """
    if not torch.cuda.is_available():
        DESCRIPTION += "\n<p>Running on CPU 🥶 This demo does not work on CPU.</p>"

    demo = gr.Interface(
        fn=generate_img,
        inputs=[Textbox(label="Note: If you want to specify a aspect ratio or determine a customized height and width, "
                              "use --ar h:w (or --aspect_ratio h:w) or --hw h:w. If no aspect ratio or hw is given, all setting will be default.",
                        placeholder="Please enter your prompt. \n"),
                gr.Radio(
                    choices=["iddpm", "dpm-solver"],
                    label=f"Sampler",
                    interactive=True,
                    value='dpm-solver',
                ),
                gr.Slider(
                    label='Sample Steps',
                    minimum=1,
                    maximum=100,
                    value=14,
                    step=1
                ),
                gr.Slider(
                    label='Guidance Scale',
                    minimum=0.1,
                    maximum=30.0,
                    value=4.5,
                    step=0.1
                ),
                gr.Slider(
                    label="Seed",
                    minimum=0,
                    maximum=MAX_SEED,
                    step=1,
                    value=0,
                ),
                gr.Checkbox(label="Randomize seed", value=True),
                ],
        outputs=[Image(type="numpy", label="Img"),
                 Textbox(label="clean prompt"),
                 Textbox(label="model info"),
                 gr.Slider(label='seed')],
        title=title,
        description=DESCRIPTION,
        examples=examples,
    )
    demo.launch(server_name="0.0.0.0", server_port=args.port, debug=True)