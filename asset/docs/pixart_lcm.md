<!--Copyright 2023 The Huawei Noah’s Ark Lab Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

<p align="center">
  <img src="https://raw.githubusercontent.com/PixArt-alpha/PixArt-alpha.github.io/master/static/images/pixart-lcm2.png"  height=120>
</p>

## 🔥 Why Need PixArt-LCM
Following [LCM LoRA](https://huggingface.co/blog/lcm_lora), we illustrative of the generation speed we achieve on various computers. Let us stress again how liberating it is to explore image generation so easily with PixArt-LCM.

| Hardware                    | PixArt-LCM (4 steps) | SDXL LoRA LCM (4 steps) | PixArt standard (14 steps) | SDXL standard (25 steps) |
|-----------------------------|----------------------|-------------------------|----------------------------|---------------------------|
| T4 (Google Colab Free Tier) | 3.3s                 | 8.4s                    | 16.0s                      | 26.5s                     |
| A100 (80 GB)                | 0.51s                | 1.2s                    | 2.2s                       | 3.8s                      |
| V100 (32 GB)                | 0.8s                 | 1.2s                    | 5.5s                       | 7.7s                      |

These tests were run with a batch size of 1 in all cases.

For cards with a lot of capacity, such as A100, performance increases significantly when generating multiple images at once, which is usually the case for production workloads.

## Training the `PixArt + LCM` on your machine

```bash
python -m torch.distributed.launch --nproc_per_node=2 --master_port=12345 train_scripts/train_pixart_lcm.py configs/pixart_config/PixArt_xl2_img1024_lcm.py --work-dir output/pixartlcm-xl2-img1024_ft
```

## Trainig the `PixArt + LCM-LoRA`

```bash
python -m torch.distributed.launch --nproc_per_node=2 --master_port=12345 train_scripts/train_pixart_lcm_lora.py configs/pixart_config/PixArt_xl2_img1024_lcm.py --work-dir output/pixartlcm-lora-xl2-img1024_ft
```

## Testing the `PixArt + LCM` on your machine

```bash
DEMO_PORT=12345 python app/app_lcm.py

Then have a look at a simple example using the http://your-server-ip:12345
```

## Testing the `PixArt + LCM-LoRA`

```bash
DEMO_PORT=12345 python app/app_lcm.py --is_lora --lora_repo_id output/pixartlcm-lora-xl2-img1024_ft/checkpoint-xxx

Then have a look at a simple example using the http://your-server-ip:12345
```

## Integration in diffusers
### Using in 🧨 diffusers

Make sure you have the updated versions of the following libraries:

```bash
pip install -U transformers accelerate diffusers
```

And then:

```python
import torch
from diffusers import PixArtAlphaPipeline, AutoencoderKL

# for PixArt-LCM
pipe = PixArtAlphaPipeline.from_pretrained("PixArt-alpha/PixArt-LCM-XL-2-1024-MS", torch_dtype=torch.float16, use_safetensors=True)

# for PixArt-LCM-LoRA
# transformer = Transformer2DModel.from_pretrained("PixArt-alpha/PixArt-LCM-XL-2-1024-MS", subfolder="transformer", torch_dtype=torch.float16)
# transformer = PeftModel.from_pretrained(transformer, "PixArt-alpha/PixArt-LCM-LoRA-XL-2-1024-MS")
# pipe = PixArtAlphaPipeline.from_pretrained("PixArt-alpha/PixArt-LCM-XL-2-1024-MS", transformer=transformer, torch_dtype=torch.float16, use_safetensors=True)
# del transformer

# Enable memory optimizations.
pipe.enable_model_cpu_offload()

prompt = "A small cactus with a happy face in the Sahara desert."
image = pipe(prompt, guidance_scale=0., num_inference_steps=4).images[0]
```

This integration allows running the pipeline with a batch size of 4 under 11 GBs of GPU VRAM. 
Check out the [documentation](https://huggingface.co/docs/diffusers/main/en/api/pipelines/pixart) to learn more.

# Keeping updating