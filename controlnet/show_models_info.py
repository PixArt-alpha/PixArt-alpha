import argparse
import sys
from pathlib import Path
from typing import Union
from pprint import pprint

current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))

import torch

# from PIL import Image
# import torchvision.transforms as T
# import torchvision.transforms.functional as TF

# from diffusion import DPMS
# from diffusion.model.hed import HEDdetector
from diffusion.model.nets import PixArtMS_XL_2, ControlPixArtHalf, ControlPixArtMSHalf

# from diffusion.model.utils import resize_and_crop_tensor
# from diffusion.utils.misc import read_config
# from diffusers.models import AutoencoderKL
# from tools.download import find_model

def pretty(d, indent=0):
    for key, value in d.items():
        # print("\t" * indent + str(key))
        print("\t" * indent + "base_model." + str(key))
        if isinstance(value, dict):
            pretty(value, indent + 1)

# the controlnet model
# model_path = "/home/raul/codelab/models/PixArt-XL-2-512-ControlNet.pth"
# model_path = "/home/raul/codelab/models/PixArt-XL-2-1024-ControlNet.pth"

# the alpha model
# model_path = "/home/raul/codelab/models/PixArt-XL-2-512x512.pth"
# model_path = "/home/raul/codelab/models/PixArt-XL-2-1024-MS.pth"

# the sigma model
# model_path = "/home/raul/codelab/models/PixArt-Sigma-XL-2-2K-MS.pth"
# model_path = "/home/raul/codelab/models/PixArt-Sigma-XL-2-1024-MS.pth"
# model_path = "/home/raul/codelab/models/PixArt-Sigma-XL-2-512-MS.pth"
model_path = "/home/raul/codelab/models/PixArt-Sigma-XL-2-256x256.pth"

model_data = torch.load(model_path)  # , map_location=lambda storage, loc: storage)

pretty(model_data['state_dict'])

# print(model_data['state_dict']['controlnet.0.after_proj.weight'])
# print(model_data['state_dict']['controlnet.0.after_proj.weight'].dtype)
# print(model_data['state_dict']['controlnet.0.after_proj.bias'])
# print(model_data['state_dict']['controlnet.0.after_proj.bias'].dtype)
