import os
from pathlib import Path
import sys
current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))
import time
import datetime
import torch.nn as nn
from PIL import Image
import torch
from torchvision import transforms as T
import numpy as np
import json
from tqdm import tqdm
import argparse

from diffusion.model.t5 import T5Embedder
from diffusers.models import AutoencoderKL


def extract_caption_t5():
    t5 = T5Embedder(device="cuda", local_cache=True, cache_dir='data/t5_ckpts')
    t5_save_root = 'data/data_toy/caption_feature_wmask'
    t5_save_dir = t5_save_root
    os.makedirs(t5_save_dir, exist_ok=True)
    captions = set()

    train_data_json = json.load(open('data/data_toy/data_info.json', 'r'))
    train_data = train_data_json[args.start_index: args.end_index]
    with torch.no_grad():
        for item in tqdm(train_data):

            caption = item['prompt'].strip()
            if caption in captions:
                continue
            captions.add(caption)
            if isinstance(caption, str):
                caption = [caption]

            save_path = os.path.join(t5_save_dir, '_'.join(item['path'].rsplit('/', 1)).replace('.png', '.npz'))
            if os.path.exists(save_path):
                continue
            try:
                caption_emb, emb_mask = t5.get_text_embeddings(caption)
                emb_dict = {
                    'caption_feature': caption_emb.float().cpu().data.numpy(),
                    'attention_mask': emb_mask.cpu().data.numpy(),
                }
                np.savez_compressed(save_path, **emb_dict)
            except Exception as e:
                print(e)

def extract_img_vae():
    vae = AutoencoderKL.from_pretrained("output/pretrained_models/sd-vae-ft-ema").to(device)

    train_data_json = json.load(open('data/data_toy/data_info.json', 'r'))
    image_names = set()

    vae_save_root = f'data/data_toy/img_vae_features_{image_resize}'
    os.umask(0o000)       # file permission: 666; dir permission: 777
    os.makedirs(vae_save_root, exist_ok=True)

    vae_save_dir = os.path.join(vae_save_root, 'noflip')
    os.makedirs(vae_save_dir, exist_ok=True)

    for item in train_data_json:
        image_name = item['path']
        if image_name in image_names:
            continue
        image_names.add(image_name)
    lines = list(image_names)
    lines.sort()
    lines = lines[args.start_index: args.end_index]

    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB')),
        T.Resize(image_resize),  # Image.BICUBIC
        T.CenterCrop(image_resize),
        T.ToTensor(),
        T.Normalize([.5], [.5]),
    ])

    os.umask(0o000)  # file permission: 666; dir permission: 777
    for image_name in tqdm(lines):
        save_path = os.path.join(vae_save_dir, image_name.replace('.jpg', '.npy'))
        if os.path.exists(save_path):
            continue
        try:
            img = Image.open(f'data/data_toy/data_imgs/{image_name}')
            img = transform(img).to(device)[None]

            with torch.no_grad():
                posterior = vae.encode(img).latent_dist
                z = torch.cat([posterior.mean, posterior.std], dim=1).detach().cpu().numpy()

            np.save(save_path, z)
            print(save_path)
        except Exception as e:
            print(e)
            print(image_name)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--multi_scale", action='store_true', default=False, help="multi-scale feature extraction")
    parser.add_argument('--start_index', default=0, type=int)
    parser.add_argument('--end_index', default=1000000, type=int)
    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not args.multi_scale:
        image_resize = 512
    else:
        raise ValueError('Multi-scale VAE feature will released soon')
    print('Extracting Image Resolution %s' % image_resize)

    # prepare extracted caption t5 features for training
    extract_caption_t5()

    # prepare extracted image vae features for training
    extract_img_vae()
