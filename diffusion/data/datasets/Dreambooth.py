from PIL import Image
import numpy as np
import torch
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS
from torch.utils.data import Dataset
from diffusers.utils.torch_utils import randn_tensor
from torchvision import transforms as T
import pathlib
from diffusers.models import AutoencoderKL

from diffusion.data.builder import get_data_path, DATASETS
from diffusion.data.datasets.utils import *

IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm', 'tif', 'tiff', 'webp', 'JPEG'}


@DATASETS.register_module()
class DreamBooth(Dataset):
    def __init__(self,
                 root,
                 transform=None,
                 resolution=1024,
                 **kwargs):
        self.root = get_data_path(root)
        path = pathlib.Path(self.root)
        self.transform = transform
        self.resolution = resolution
        self.img_samples = sorted(
            [file for ext in IMAGE_EXTENSIONS for file in path.glob(f'*.{ext}')]
        )
        self.ori_imgs_nums = len(self)
        self.loader = default_loader
        self.base_size = int(kwargs['aspect_ratio_type'].split('_')[-1])
        self.aspect_ratio = eval(kwargs.pop('aspect_ratio_type'))       # base aspect ratio
        self.ratio_nums = {}
        for k, v in self.aspect_ratio.items():
            self.ratio_nums[float(k)] = 0      # used for batch-sampler
        self.data_info = {'img_hw': torch.tensor([resolution, resolution], dtype=torch.float32), 'aspect_ratio': 1.}

        # image related
        with torch.inference_mode():
            vae = AutoencoderKL.from_pretrained("output/pretrained_models/sd-vae-ft-ema")
            imgs = []
            for img_path in self.img_samples:
                img = self.loader(img_path)
                self.ratio_nums[1.0] += 1
                if self.transform is not None:
                    imgs.append(self.transform(img))
            imgs = torch.stack(imgs, dim=0)
            self.img_vae = vae.encode(imgs).latent_dist.sample()
            del vae

    def __getitem__(self, index):
        return self.img_vae[index], self.data_info

    @staticmethod
    def vae_feat_loader(path):
        # [mean, std]
        mean, std = torch.from_numpy(np.load(path)).chunk(2)
        sample = randn_tensor(mean.shape, generator=None, device=mean.device, dtype=mean.dtype)
        return mean + std * sample

    def load_ori_img(self, img_path):
        # 加载图像并转换为Tensor
        transform = T.Compose([
            T.Resize(256),  # Image.BICUBIC
            T.CenterCrop(256),
            T.ToTensor(),
        ])
        return transform(Image.open(img_path))

    def __len__(self):
        return len(self.img_samples)

    def __getattr__(self, name):
        if name == "set_epoch":
            return lambda epoch: None
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def get_data_info(self, idx):
        return {'height': self.resolution, 'width': self.resolution}
