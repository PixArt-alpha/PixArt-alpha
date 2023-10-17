<p align="center">
  <img src="asset/logo.png"  height=120>
</p>


### <div align="center">👉 PixArt-α: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis <div> 

<div align="center">

[![Huggingface PixArt-alpha](https://img.shields.io/static/v1?label=Demo&message=Huggingface%20Gradio&color=orange)](https://huggingface.co/spaces/PixArt-alpha/PixArt) &ensp; [![Code PixArt-alpha](https://img.shields.io/static/v1?label=Code&message=Github&color=blue)](https://github.com/PixArt-alpha/PixArt) &ensp; [![arXiv](https://img.shields.io/badge/arXiv-2310.00426-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2310.00426)

</div> 

---

This is a PyTorch implementation of the following paper:

> [**PixArt-α: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis**](https://arxiv.org/abs/2310.00426)<br>
> Junsong Chen*, Jincheng Yu*, Chongjian Ge*, Lewei Yao*, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, Zhenguo Li
> <br>Huawei Noah's Ark Lab<br>

## 🐱 Abstract
<b>TL; DR: <font color="red">PixArt-α</font> is a Transformer-based T2I diffusion model whose image generation quality is competitive with state-of-the-art image generators (e.g., Imagen, SDXL, and even Midjourney), and the training speed markedly surpasses existing large-scale T2I models, e.g., PixArt-α only takes 10.8% of Stable Diffusion v1.5's training time (675 vs. 6,250 A100 GPU days).</b>

<details><summary>CLICK for the full abstract</summary>
The most advanced text-to-image (T2I) models require significant training costs (e.g., millions of GPU hours), seriously hindering the fundamental innovation for the AIGC community while increasing CO2 emissions. This paper introduces PixArt-α, a Transformer-based T2I diffusion model whose image generation quality is competitive with state-of-the-art image generators (e.g., Imagen, SDXL, and even Midjourney), reaching near-commercial application standards. Additionally, it supports high-resolution image synthesis up to 1024px resolution with low training cost. To achieve this goal, three core designs are proposed: (1) Training strategy decomposition: We devise three distinct training steps that separately optimize pixel dependency, text-image alignment, and image aesthetic quality; (2) Efficient T2I Transformer: We incorporate cross-attention modules into Diffusion Transformer (DiT) to inject text conditions and streamline the computation-intensive class-condition branch; (3) High-informative data: We emphasize the significance of concept density in text-image pairs and leverage a large Vision-Language model to auto-label dense pseudo-captions to assist text-image alignment learning. As a result, PixArt-α's training speed markedly surpasses existing large-scale T2I models, e.g., PixArt-α only takes 10.8% of Stable Diffusion v1.5's training time (675 vs. 6,250 A100 GPU days), saving nearly $300,000 ($26,000 vs. $320,000) and reducing 90% CO2 emissions. Moreover, compared with a larger SOTA model, RAPHAEL, our training cost is merely 1%. Extensive experiments demonstrate that PixArt-α excels in image quality, artistry, and semantic control. We hope PixArt-α will provide new insights to the AIGC community and startups to accelerate building their own high-quality yet low-cost generative models from scratch.
</details>

---

![A small cactus with a happy face in the Sahara desert.](asset/images/teaser.png)

---



## 🚩 **New Features/Updates**
- ✅ Oct. 15, 2023. Release the inference code and pretrained model of [PixArt-α](https://github.com/PixArt-alpha/PixArt).

---

# 🔥🔥🔥 Why PixArt-α? 
## Training Efficiency
PixArt-α only takes 10.8% of Stable Diffusion v1.5's training time (675 vs. 6,250 A100 GPU days), saving nearly $300,000 ($26,000 vs. $320,000) and reducing 90% CO2 emissions. Moreover, compared with a larger SOTA model, RAPHAEL, our training cost is merely 1%.
![Training Efficiency.](asset/images/efficiency.svg)


## High-quality Generation from PixArt-α.

- More samples
<div id="dreambooth" style="display: flex; justify-content: center;">
  <img src="asset/images/more-samples1.png" style="width: 49%; height: auto; object-fit: contain; margin: 5px;">
  <img src="asset/images/more-samples.png" style="width: 49%; height: auto; object-fit: contain; margin: 5px;">
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
All models will be automatically downloaded. You can also choose to download manually from this [url](https://huggingface.co/PixArt-alpha/PixArt).

# 💻 How to Test
Inference requires at least `23GB` of GPU memory.

## Quick start with [Gradio](https://www.gradio.app/guides/quickstart)

To get started, first install the required dependencies:

```bash
python scripts/interface.py --model_path path/to/model.pth --image_size=1024 --port=12345
```
Let's have a look at a simple example using the `http://your-server-ip:port`.

## Online Demo [![Huggingface PixArt](https://img.shields.io/static/v1?label=Demo&message=Huggingface%20Gradio&color=orange)](https://huggingface.co/spaces/PixArt-alpha/PixArt) 
![Training Efficiency.](asset/images/sample.png)

...

## 🔥To-Do List

- [x] inference code & model
- [ ] diffusers version
- [ ] training code


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
