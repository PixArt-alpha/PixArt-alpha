import os
import numpy as np
import torch
import random
from torchvision.datasets.folder import default_loader
from diffusion.data.datasets.InternalData import InternalData
from diffusion.data.builder import get_data_path, DATASETS
from diffusion.utils.logger import get_root_logger
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from diffusion.data.datasets.utils import *

def get_closest_ratio(height: float, width: float, ratios: dict):
    aspect_ratio = height / width
    closest_ratio = min(ratios.keys(), key=lambda ratio: abs(float(ratio) - aspect_ratio))
    return ratios[closest_ratio], float(closest_ratio)


@DATASETS.register_module()
class InternalDataMS(InternalData):
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
                 mask_type='null',
                 load_mask_index=False,
                 max_length=120,
                 config=None,
                 **kwargs):
        self.root = get_data_path(root)
        self.transform = transform
        self.load_vae_feat = load_vae_feat
        self.ori_imgs_nums = 0
        self.resolution = resolution
        self.N = int(resolution // (input_size // patch_size))
        self.mask_ratio = mask_ratio
        self.load_mask_index = load_mask_index
        self.mask_type = mask_type
        self.base_size = int(kwargs['aspect_ratio_type'].split('_')[-1])
        self.max_lenth = max_length
        self.aspect_ratio = eval(kwargs.pop('aspect_ratio_type'))       # base aspect ratio
        self.meta_data_clean = []
        self.img_samples = []
        self.txt_feat_samples = []
        self.vae_feat_samples = []
        self.mask_index_samples = []
        self.ratio_index = {}
        self.ratio_nums = {}
        for k, v in self.aspect_ratio.items():
            self.ratio_index[float(k)] = []     # used for self.getitem
            self.ratio_nums[float(k)] = 0      # used for batch-sampler

        image_list_json = image_list_json if isinstance(image_list_json, list) else [image_list_json]
        for json_file in image_list_json:
            meta_data = self.load_json(os.path.join(self.root, 'partition_filter', json_file))
            self.ori_imgs_nums += len(meta_data)
            meta_data_clean = [item for item in meta_data if item['ratio'] <= 4]
            self.meta_data_clean.extend(meta_data_clean)
            self.img_samples.extend([os.path.join(self.root.replace('InternData', "InternImgs"), item['path']) for item in meta_data_clean])
            self.txt_feat_samples.extend([os.path.join(self.root, 'caption_feature_wmask', '_'.join(item['path'].rsplit('/', 1)).replace('.png', '.npz')) for item in meta_data_clean])
            self.vae_feat_samples.extend([os.path.join(self.root, f'img_vae_fatures_{resolution}_multiscale/ms', '_'.join(item['path'].rsplit('/', 1)).replace('.png', '.npy')) for item in meta_data_clean])

        # Set loader and extensions
        if load_vae_feat:
            self.transform = None
            self.loader = self.vae_feat_loader
        else:
            self.loader = default_loader

        if sample_subset is not None:
            self.sample_subset(sample_subset)  # sample dataset for local debug

        # scan the dataset for ratio static
        for i, info in enumerate(self.meta_data_clean[:len(self.meta_data_clean)//3]):
            ori_h, ori_w = info['height'], info['width']
            closest_size, closest_ratio = get_closest_ratio(ori_h, ori_w, self.aspect_ratio)
            self.ratio_nums[closest_ratio] += 1
            if len(self.ratio_index[closest_ratio]) == 0:
                self.ratio_index[closest_ratio].append(i)
        # print(self.ratio_nums)
        logger = get_root_logger() if config is None else get_root_logger(os.path.join(config.work_dir, 'train_log.log'))
        logger.info(f"T5 max token length: {self.max_lenth}")

    def getdata(self, index):
        img_path = self.img_samples[index]
        npz_path = self.txt_feat_samples[index]
        npy_path = self.vae_feat_samples[index]
        ori_h, ori_w = self.meta_data_clean[index]['height'], self.meta_data_clean[index]['width']

        # Calculate the closest aspect ratio and resize & crop image[w, h]
        closest_size, closest_ratio = get_closest_ratio(ori_h, ori_w, self.aspect_ratio)
        closest_size = list(map(lambda x: int(x), closest_size))
        self.closest_ratio = closest_ratio

        if self.load_vae_feat:
            try:
                img = self.loader(npy_path)
                if index not in self.ratio_index[closest_ratio]:
                    self.ratio_index[closest_ratio].append(index)
            except Exception:
                index = random.choice(self.ratio_index[closest_ratio])
                return self.getdata(index)
            h, w = (img.shape[1], img.shape[2])
            assert h, w == (ori_h//8, ori_w//8)
        else:
            img = self.loader(img_path)
            h, w = (img.size[1], img.size[0])
            assert h, w == (ori_h, ori_w)

        data_info = {'img_hw': torch.tensor([ori_h, ori_w], dtype=torch.float32)}
        data_info['aspect_ratio'] = closest_ratio
        data_info["mask_type"] = self.mask_type

        txt_info = np.load(npz_path)
        txt_fea = torch.from_numpy(txt_info['caption_feature'])
        attention_mask = torch.ones(1, 1, txt_fea.shape[1])
        if 'attention_mask' in txt_info.keys():
            attention_mask = torch.from_numpy(txt_info['attention_mask'])[None]

        if not self.load_vae_feat:
            if closest_size[0] / ori_h > closest_size[1] / ori_w:
                resize_size = closest_size[0], int(ori_w * closest_size[0] / ori_h)
            else:
                resize_size = int(ori_h * closest_size[1] / ori_w), closest_size[1]
            self.transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB')),
                T.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),  # Image.BICUBIC
                T.CenterCrop(closest_size),
                T.ToTensor(),
                T.Normalize([.5], [.5]),
            ])

        if self.transform:
            img = self.transform(img)

        return img, txt_fea, attention_mask, data_info

    def __getitem__(self, idx):
        for _ in range(20):
            try:
                return self.getdata(idx)
            except Exception as e:
                print(f"Error details: {str(e)}")
                idx = random.choice(self.ratio_index[self.closest_ratio])
        raise RuntimeError('Too many bad data.')
