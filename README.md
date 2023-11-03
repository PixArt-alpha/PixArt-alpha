<p align="center">
  <img src="asset/logo.png"  height=120>
</p>


### <div align="center">👉 PixArt-α: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis <div> 

<div align="center">

[![Huggingface PixArt-alpha](https://img.shields.io/static/v1?label=Demo&message=Huggingface%20Gradio&color=orange)](https://huggingface.co/spaces/PixArt-alpha/PixArt-alpha) &ensp; [![Project page PixArt-alpha](https://img.shields.io/static/v1?label=Project&message=Github&color=blue)](https://pixart-alpha.github.io/) &ensp; [![arXiv](https://img.shields.io/badge/arXiv-2310.00426-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2310.00426)

</div> 

---

This repo contains PyTorch model definitions, pre-trained weights and inference/sampling code for our paper exploring 
Fast training diffusion models with transformers. You can find more visualizations on our [project page](https://pixart-alpha.github.io/).

> [**PixArt-α: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis**](https://pixart-alpha.github.io/)<br>
> [Junsong Chen*](https://lawrence-cj.github.io/), [Jincheng Yu*](https://lovesykun.cn/about.html), 
> [Chongjian Ge*](https://chongjiange.github.io/), [Lewei Yao*](https://scholar.google.com/citations?user=hqDyTg8AAAAJ&hl=zh-CN&oi=ao),
> [Enze Xie](https://xieenze.github.io/)&#8224;,
> [Yue Wu](https://yuewuhkust.github.io/), [Zhongdao Wang](https://zhongdao.github.io/), 
> [James Kwok](https://www.cse.ust.hk/~jamesk/), [Ping Luo](http://luoping.me/), 
> [Huchuan Lu](https://scholar.google.com/citations?hl=en&user=D3nE0agAAAAJ), 
> [Zhenguo Li](https://scholar.google.com/citations?user=XboZC1AAAAAJ)
> <br>Huawei Noah’s Ark Lab, Dalian University of Technology, HKU, HKUST<br>

---

## 🚩 **New Features/Updates**

- ✅ Oct. 27, 2023. Release the training & feature extraction code.
- ✅ Oct. 20, 2023. Collaborate with Hugging Face & Diffusers team to co-release the code and weights. (plz stay tuned.)
- ✅ Oct. 15, 2023. Release the inference code.

---

## 🐱 Abstract
<b>TL; DR: <font color="red">PixArt-α</font> is a Transformer-based T2I diffusion model whose image generation quality is competitive with state-of-the-art image generators (e.g., Imagen, SDXL, and even Midjourney), and the training speed markedly surpasses existing large-scale T2I models, e.g., PixArt-α only takes 10.8% of Stable Diffusion v1.5's training time (675 vs. 6,250 A100 GPU days).</b>

<details><summary>CLICK for the full abstract</summary>
The most advanced text-to-image (T2I) models require significant training costs (e.g., millions of GPU hours), 
seriously hindering the fundamental innovation for the AIGC community while increasing CO2 emissions. 
This paper introduces PixArt-α, a Transformer-based T2I diffusion model whose image generation quality is competitive with state-of-the-art image generators (e.g., Imagen, SDXL, and even Midjourney), 
reaching near-commercial application standards. Additionally, it supports high-resolution image synthesis up to 1024px resolution with low training cost. 
To achieve this goal, three core designs are proposed: 
(1) Training strategy decomposition: We devise three distinct training steps that separately optimize pixel dependency, text-image alignment, and image aesthetic quality; 
(2) Efficient T2I Transformer: We incorporate cross-attention modules into Diffusion Transformer (DiT) to inject text conditions and streamline the computation-intensive class-condition branch; 
(3) High-informative data: We emphasize the significance of concept density in text-image pairs and leverage a large Vision-Language model to auto-label dense pseudo-captions to assist text-image alignment learning. 
As a result, PixArt-α's training speed markedly surpasses existing large-scale T2I models, 
e.g., PixArt-α only takes 10.8% of Stable Diffusion v1.5's training time (675 vs. 6,250 A100 GPU days), 
saving nearly $300,000 ($26,000 vs. $320,000) and reducing 90% CO2 emissions. Moreover, compared with a larger SOTA model, RAPHAEL, 
our training cost is merely 1%. Extensive experiments demonstrate that PixArt-α excels in image quality, artistry, and semantic control. 
We hope PixArt-α will provide new insights to the AIGC community and startups to accelerate building their own high-quality yet low-cost generative models from scratch.
</details>

---

![A small cactus with a happy face in the Sahara desert.](asset/images/teaser.png)

---

# 🔥🔥🔥 Why PixArt-α? 
## Training Efficiency
PixArt-α only takes 10.8% of Stable Diffusion v1.5's training time (675 vs. 6,250 A100 GPU days), saving nearly $300,000 ($26,000 vs. $320,000) and reducing 90% CO2 emissions. Moreover, compared with a larger SOTA model, RAPHAEL, our training cost is merely 1%.
![Training Efficiency.](asset/images/efficiency.svg)

| Method    | Type | #Params | #Images | A100 GPU days |
|-----------|------|---------|---------|---------------|
| DALL·E    | Diff | 12.0B   | 1.54B   |               |
| GLIDE     | Diff | 5.0B    | 5.94B   |               |
| LDM       | Diff | 1.4B    | 0.27B   |               |
| DALL·E 2  | Diff | 6.5B    | 5.63B   | 41,66         |
| SDv1.5    | Diff | 0.9B    | 3.16B   | 6,250         |
| GigaGAN   | GAN  | 0.9B    | 0.98B   | 4,783         |
| Imagen    | Diff | 3.0B    | 15.36B  | 7,132         |
| RAPHAEL   | Diff | 3.0B    | 5.0B    | 60,000        |
| PixArt-α  | Diff | 0.6B    | 0.025B  | 675           |

## High-quality Generation from PixArt-α

- More samples
<div id="dreambooth" style="display: flex; justify-content: center;">
  <img src="asset/images/more-samples1.png" style="width: 50%; height: auto; object-fit: contain; margin: 5px;">
  <img src="asset/images/more-samples.png" style="width: 43%; height: auto; object-fit: contain; margin: 5px;">
</div>

- PixArt + [Dreambooth](https://dreambooth.github.io/)
<div id="dreambooth" style="display: flex; justify-content: center;">
  <img src="asset/images/dreambooth/dreambooth_dog.svg" width="46%" style="margin: 5px;">
  <img src="asset/images/dreambooth/dreambooth_m5.svg" width="46%" style="margin: 5px;">
</div>

- PixArt + [ControlNet](https://github.com/lllyasviel/ControlNet)
<div id="dreambooth" style="display: flex; justify-content: center;">
  <img src="asset/images/controlnet/controlnet_huawei.svg" width="46%" style="margin: 5px;">
  <img src="asset/images/controlnet/controlnet_lenna.svg" width="46%" style="margin: 5px;">
</div>

# 🔧 Dependencies and Installation

- Python >= 3.10 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.13.0+cu11.7](https://pytorch.org/)
```bash
conda create -n pixart python==3.9.0
conda activate pixart
cd path/to/pixart
pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```

# ⏬ Download Models
All models will be automatically downloaded. You can also choose to download manually from this [url](https://huggingface.co/PixArt-alpha/PixArt-alpha).

| Model            | #Params  | url                                                                             |
|:-----------------|:---------|:--------------------------------------------------------------------------------|
| T5               | 4.3B     | [T5](https://huggingface.co/PixArt-alpha/PixArt-alpha/tree/main/t5-v1_1-xxl)    |
| VAE              | 80M      | [VAE](https://huggingface.co/PixArt-alpha/PixArt-alpha/tree/main/sd-vae-ft-ema) |
| PixArt-α-SAM-256 | 0.6B     | [256](about:blank)                                                              |
| PixArt-α-512     | 0.6B     | [512](about:blank)                                                              |
| PixArt-α-1024    | 0.6B     | [1024](about:blank)                                                             |

# 🔥 How to Train
Here we take SAM dataset training config as an example, but of course, you can also prepare your own dataset following this method.

You **ONLY** need to change the **config** file in [config](./configs/pixart_config) and **dataloader** in [dataset](./diffusion/data/datasets).
```bash
python -m torch.distributed.launch --nproc_per_node=2 --master_port=12345 scripts/train.py configs/pixart_config/PixArt_xl2_img256_SAM.py --work-dir output/train_SAM_256
```

The directory structure for SAM dataset is:
```
cd ./data

SA1B
├──images/
│  ├──sa_xxxxx.jpg
│  ├──sa_xxxxx.jpg
│  ├──......
├──partition/
│  ├──part0.txt
│  ├──part1.txt
│  ├──......
├──caption_feature_wmask/
│  ├──sa_xxxxx.npz
│  ├──sa_xxxxx.npz
│  ├──......
├──img_vae_feature/
│  ├──train_vae_256/
│  │  ├──noflip/
│  │  │  ├──sa_xxxxx.npy
│  │  │  ├──sa_xxxxx.npy
│  │  │  ├──......

```


# 💻 How to Test
Inference requires at least `23GB` of GPU memory.
Currently support:
- [x] [IDDPM](https://arxiv.org/abs/2102.09672)
- [x] [DPM-Solver](https://arxiv.org/abs/2206.00927)
- [x] [SA-Solver (waiting for official release)](https://arxiv.org/abs/2309.05019)
- [ ] [DPM-Solver-v3](https://arxiv.org/abs/2310.13268v2)

## Quick start with [Gradio](https://www.gradio.app/guides/quickstart)

To get started, first install the required dependencies, then run:

```bash
python scripts/interface.py --model_path path/to/model.pth --image_size=1024 --port=12345
```
Let's have a look at a simple example using the `http://your-server-ip:port`.


## Online Demo [![Huggingface PixArt](https://img.shields.io/static/v1?label=Demo&message=Huggingface%20Gradio&color=orange)](https://huggingface.co/spaces/PixArt-alpha/PixArt-alpha) 
![Training Efficiency.](asset/images/sample.png)


## 🔥To-Do List

- [x] Inference code
- [x] T2ICompBench code
- [x] Training code
- [x] T5 & VAE feature extraction code
- [ ] Model zoo 
- [ ] Diffusers version

[//]: # (https://user-images.githubusercontent.com/73707470/253800159-c7e12362-1ea1-4b20-a44e-bd6c8d546765.mp4)


# 📖BibTeX
    @misc{chen2023pixartalpha,
          title={PixArt-$\alpha$: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis}, 
          author={Junsong Chen and Jincheng Yu and Chongjian Ge and Lewei Yao and Enze Xie and Yue Wu and Zhongdao Wang and James Kwok and Ping Luo and Huchuan Lu and Zhenguo Li},
          year={2023},
          eprint={2310.00426},
          archivePrefix={arXiv},
          primaryClass={cs.CV}
    }
    
# 🤗Acknowledgements
- Thanks to [DiT](https://github.com/facebookresearch/DiT) for their wonderful work and codebase.
