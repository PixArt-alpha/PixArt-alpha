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


# model_path = "/home/raul/codelab/models/PixArt-XL-2-512-ControlNet.pth"
# model_path = "/home/raul/codelab/models/PixArt-XL-2-1024-ControlNet.pth"
# model_path = "/home/raul/codelab/models/PixArt-XL-2-512x512.pth"
model_path = "/home/raul/codelab/models/PixArt-XL-2-1024-MS.pth"
model_data = torch.load(model_path)  # , map_location=lambda storage, loc: storage)

pretty(model_data['state_dict'])

# if config.image_size == 512:
#     model = PixArt_XL_2(
#         input_size=latent_size, lewei_scale=lewei_scale[config.image_size]
#     )
#     print("model architecture ControlPixArtHalf and image size is 512")
#     model = ControlPixArtHalf(model).to(device)
# elif config.image_size == 1024:
#     model = PixArtMS_XL_2(
#         input_size=latent_size, lewei_scale=lewei_scale[config.image_size]
#     )
#     print("model architecture ControlPixArtMSHalf and image size is 1024")
#     model = ControlPixArtMSHalf(model).to(device)
