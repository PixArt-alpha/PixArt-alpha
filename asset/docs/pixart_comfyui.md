<!--Copyright 2023 The Huawei Noah’s Ark Lab Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

## 🔥 How to use PixArt in ComfyUI

### 1. Preparation for PixArt running envrironment

```bash
cd /workspace

conda create -n pixart python==3.9.0
conda activate pixart
pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117

git clone https://github.com/PixArt-alpha/PixArt-alpha.git
cd PixArt-alpha
pip install -r requirements.txt
```

### 2. Install ComfyUI related dependencies

```bash
cd /workspace
git clone https://github.com/comfyanonymous/ComfyUI.git

cd ComfyUI
git clone https://github.com/city96/ComfyUI_ExtraModels custom_nodes/ComfyUI_ExtraModels
```

### 3. Download all the checkpoints: PixArt, VAE, T5 with script

```bash
cd /workspace/PixArt
python tools/download.py --model_names "PixArt-XL-2-1024-MS.pth"
```
or download with urls:[PixArt ckpt](https://huggingface.co/PixArt-alpha/PixArt-alpha/resolve/main/PixArt-XL-2-1024-MS.pth), [VAE ckpt](https://huggingface.co/PixArt-alpha/PixArt-alpha/tree/main/sd-vae-ft-ema), 
[T5 ckpt](https://huggingface.co/PixArt-alpha/PixArt-alpha/tree/main/t5-v1_1-xxl).

### 4. Put Checkpoints into corresponding folders
```bash
cd /workspace/ComfyUI

mv /path/to/PixArt-XL-2-1024-MS.pth ./models/checkpoints/
mv /path/to/sd-vae-ft-ema ./models/VAE/
mv /path/to/t5-v1_1-xxl ./models/t5/
```
### 5. run the ComfyUI website
```bash
cd /workspace/ComfyUI

python main.py --port 11111 --listen 0.0.0.0
```
Open http://your-server-ip:11111 to play with PixArt.

### 6. Create your own custom nodes
Here we prepare two examples for better understanding:

1) [PixArt Text-to-Image workflow](https://huggingface.co/PixArt-alpha/PixArt-alpha/blob/main/PixArt-image-to-image-workflow.json)

2) [PixArt Image-to-Image workflow](https://huggingface.co/PixArt-alpha/PixArt-alpha/blob/main/PixArt-image-to-image-workflow.json)

Once you download these json files, you can open your server website which is `http://your-server-ip:11111` and drop the json file into the website window to begin the PixArt-ComfyUI playground.