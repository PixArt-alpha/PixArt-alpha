# This is an improved version and model of HED edge detection with Apache License, Version 2.0.
# Please use this implementation in your products
# This implementation may produce slightly different results from Saining Xie's official implementations,
# but it generates smoother edges and is more suitable for ControlNet as well as other image-to-image translations.
# Different from official models and other implementations, this is an RGB-input model (rather than BGR)
# and in this way it works better for gradio's RGB protocol
import sys
from pathlib import Path
current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent.parent))
from torch import nn
import torch
import numpy as np
from torchvision import transforms as T
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import json
from PIL import Image
import torchvision.transforms.functional as TF
from accelerate import Accelerator
from diffusers.models import AutoencoderKL
import os

image_resize = 1024


class DoubleConvBlock(nn.Module):
    def __init__(self, input_channel, output_channel, layer_number):
        super().__init__()
        self.convs = torch.nn.Sequential()
        self.convs.append(torch.nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=(3, 3), stride=(1, 1), padding=1))
        for i in range(1, layer_number):
            self.convs.append(torch.nn.Conv2d(in_channels=output_channel, out_channels=output_channel, kernel_size=(3, 3), stride=(1, 1), padding=1))
        self.projection = torch.nn.Conv2d(in_channels=output_channel, out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding=0)

    def forward(self, x, down_sampling=False):
        h = x
        if down_sampling:
            h = torch.nn.functional.max_pool2d(h, kernel_size=(2, 2), stride=(2, 2))
        for conv in self.convs:
            h = conv(h)
            h = torch.nn.functional.relu(h)
        return h, self.projection(h)


class ControlNetHED_Apache2(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = torch.nn.Parameter(torch.zeros(size=(1, 3, 1, 1)))
        self.block1 = DoubleConvBlock(input_channel=3, output_channel=64, layer_number=2)
        self.block2 = DoubleConvBlock(input_channel=64, output_channel=128, layer_number=2)
        self.block3 = DoubleConvBlock(input_channel=128, output_channel=256, layer_number=3)
        self.block4 = DoubleConvBlock(input_channel=256, output_channel=512, layer_number=3)
        self.block5 = DoubleConvBlock(input_channel=512, output_channel=512, layer_number=3)

    def forward(self, x):
        h = x - self.norm
        h, projection1 = self.block1(h)
        h, projection2 = self.block2(h, down_sampling=True)
        h, projection3 = self.block3(h, down_sampling=True)
        h, projection4 = self.block4(h, down_sampling=True)
        h, projection5 = self.block5(h, down_sampling=True)
        return projection1, projection2, projection3, projection4, projection5


class InternData(Dataset):
    def __init__(self):
        ####
        with open('data/InternData/partition/data_info.json', 'r') as f:
            self.j = json.load(f)
        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB')),
            T.Resize(image_resize),  # Image.BICUBIC
            T.CenterCrop(image_resize),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.j)

    def getdata(self, idx):

        path = self.j[idx]['path']
        image = Image.open("data/InternImgs/" + path)
        image = self.transform(image)
        return image, path

    def __getitem__(self, idx):
        for i in range(20):
            try:
                data = self.getdata(idx)
                return data
            except Exception as e:
                print(f"Error details: {str(e)}")
                idx = np.random.randint(len(self))
        raise RuntimeError('Too many bad data.')

class HEDdetector(nn.Module):
    def __init__(self, feature=True, vae=None):
        super().__init__()
        self.model = ControlNetHED_Apache2()
        self.model.load_state_dict(torch.load('output/pretrained_models/ControlNetHED.pth', map_location='cpu'))
        self.model.eval()
        self.model.requires_grad_(False)
        if feature:
            if vae is None:
                self.vae = AutoencoderKL.from_pretrained("output/pretrained_models/sd-vae-ft-ema")
            else:
                self.vae = vae
            self.vae.eval()
            self.vae.requires_grad_(False)
        else:
            self.vae = None

    def forward(self, input_image):
        B, C, H, W = input_image.shape
        with torch.inference_mode():
            edges = self.model(input_image * 255.)
            edges = torch.cat([TF.resize(e, [H, W]) for e in edges], dim=1)
            edge = 1 / (1 + torch.exp(-torch.mean(edges, dim=1, keepdim=True)))
            edge.clip_(0, 1)
            if self.vae:
                edge = TF.normalize(edge, [.5], [.5])
                edge = edge.repeat(1, 3, 1, 1)
                posterior = self.vae.encode(edge).latent_dist
                edge = torch.cat([posterior.mean, posterior.std], dim=1).cpu().numpy()
        return edge


def main():
    dataset = InternData()
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=8, pin_memory=True)
    hed = HEDdetector()

    accelerator = Accelerator()
    hed, dataloader = accelerator.prepare(hed, dataloader)


    for img, path in tqdm(dataloader):
        out = hed(img.cuda())
        for p, o in zip(path, out):
            save = f'data/InternalData/hed_feature_{image_resize}/' + p.replace('.png', '.npz')
            if os.path.exists(save):
                continue
            os.makedirs(os.path.dirname(save), exist_ok=True)
            np.savez_compressed(save, o)


if __name__ == "__main__":
    main()
