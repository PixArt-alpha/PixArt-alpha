import os
import random
from PIL import Image
import numpy as np
import torch
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS
from torch.utils.data import Dataset
from diffusers.utils.torch_utils import randn_tensor
from torchvision import transforms as T
from diffusion.data.builder import get_data_path, DATASETS

import json, time


@DATASETS.register_module()
class InternalDataHed(Dataset):
    def __init__(self,
                 root,
                 image_list_json='data_info.json',
                 transform=None,
                 resolution=256,
                 sample_subset=None,
                 load_vae_feat=False,
                 input_size=32,
                 patch_size=2,
                 mask_ratio=0.0,
                 load_mask_index=False,
                 train_ratio=1.0,
                 mode='train',
                 **kwargs):
        self.root = get_data_path(root)
        self.transform = transform
        self.load_vae_feat = load_vae_feat
        self.ori_imgs_nums = 0
        self.resolution = resolution
        self.N = int(resolution // (input_size // patch_size))
        self.mask_ratio = mask_ratio
        self.load_mask_index = load_mask_index
        self.meta_data_clean = []
        self.img_samples = []
        self.txt_feat_samples = []
        self.vae_feat_samples = []
        self.hed_feat_samples = []
        self.prompt_samples = []

        image_list_json = image_list_json if isinstance(image_list_json, list) else [image_list_json]
        for json_file in image_list_json:
            meta_data = self.load_json(os.path.join(self.root, 'partition_filter', json_file))
            self.ori_imgs_nums += len(meta_data)
            meta_data_clean = [item for item in meta_data if item['ratio'] <= 4]
            self.meta_data_clean.extend(meta_data_clean)
            self.img_samples.extend([os.path.join(self.root.replace('InternData', "InternImgs"), item['path']) for item in meta_data_clean])
            self.txt_feat_samples.extend([os.path.join(self.root, 'caption_features', '_'.join(item['path'].rsplit('/', 1)).replace('.png', '.npz')) for item in meta_data_clean])
            self.vae_feat_samples.extend([os.path.join(self.root, f'img_vae_features_{resolution}resolution/noflip', '_'.join(item['path'].rsplit('/', 1)).replace('.png', '.npy')) for item in meta_data_clean])
            self.hed_feat_samples.extend([os.path.join(self.root, f'hed_feature_{resolution}', item['path'].replace('.png', '.npz')) for item in meta_data_clean])
            self.prompt_samples.extend([item['prompt'] for item in meta_data_clean])

        total_sample = len(self.img_samples)
        used_sample_num = int(total_sample * train_ratio)
        print("using mode", mode)
        if mode == 'train':
            self.img_samples = self.img_samples[:used_sample_num]
            self.txt_feat_samples = self.txt_feat_samples[:used_sample_num]
            self.vae_feat_samples = self.vae_feat_samples[:used_sample_num]
            self.hed_feat_samples = self.hed_feat_samples[:used_sample_num]
            self.prompt_samples = self.prompt_samples[:used_sample_num]
        else:
            self.img_samples = self.img_samples[-used_sample_num:]
            self.txt_feat_samples = self.txt_feat_samples[-used_sample_num:]
            self.vae_feat_samples = self.vae_feat_samples[-used_sample_num:]
            self.hed_feat_samples = self.hed_feat_samples[-used_sample_num:]
            self.prompt_samples = self.prompt_samples[-used_sample_num:]

        # Set loader and extensions
        if load_vae_feat:
            self.transform = None
            self.loader = self.vae_feat_loader
        else:
            self.loader = default_loader

        if sample_subset is not None:
            self.sample_subset(sample_subset)  # sample dataset for local debug

    def getdata(self, index):
        img_path = self.img_samples[index]
        npz_path = self.txt_feat_samples[index]
        npy_path = self.vae_feat_samples[index]
        hed_npz_path = self.hed_feat_samples[index]
        prompt = self.prompt_samples[index]
        # only trained on single-scale 1024 res data
        data_info = {'img_hw': torch.tensor([1024., 1024.], dtype=torch.float32), 'aspect_ratio': torch.tensor(1.)}

        if self.load_vae_feat:
            img = self.loader(npy_path)
        else:
            img = self.loader(img_path)
        hed_fea = self.vae_feat_loader_npz(hed_npz_path)
        txt_info = np.load(npz_path)
        txt_fea = torch.from_numpy(txt_info['caption_feature'])
        attention_mask = torch.ones(1, 1, txt_fea.shape[1])
        if 'attention_mask' in txt_info.keys():
            attention_mask = torch.from_numpy(txt_info['attention_mask'])[None]

        if self.transform:
            img = self.transform(img)

        data_info['condition'] = hed_fea
        data_info['prompt'] = prompt
        return img, txt_fea, attention_mask, data_info

    def __getitem__(self, idx):
        for i in range(20):
            try:
                data = self.getdata(idx)
                return data
            except Exception as e:
                print(f"Error details: {str(e)}")
                idx = np.random.randint(len(self))
        raise RuntimeError('Too many bad data.')

    def get_data_info(self, idx):
        data_info = self.meta_data_clean[idx]
        return {'height': data_info['height'], 'width': data_info['width']}

    @staticmethod
    def vae_feat_loader(path):
        # [mean, std]
        mean, std = torch.from_numpy(np.load(path)).chunk(2)
        sample = randn_tensor(mean.shape, generator=None, device=mean.device, dtype=mean.dtype)
        return mean + std * sample

    @staticmethod
    def vae_feat_loader_npz(path):
        # [mean, std]
        mean, std = torch.from_numpy(np.load(path)['arr_0']).chunk(2)
        sample = randn_tensor(mean.shape, generator=None, device=mean.device, dtype=mean.dtype)
        return mean + std * sample

    def load_json(self, file_path):
        with open(file_path, 'r') as f:
            meta_data = json.load(f)

        return meta_data

    def sample_subset(self, ratio):
        sampled_idx = random.sample(list(range(len(self))), int(len(self) * ratio))
        self.img_samples = [self.img_samples[i] for i in sampled_idx]

    def __len__(self):
        return len(self.img_samples)

    def __getattr__(self, name):
        if name == "set_epoch":
            return lambda epoch: None
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
