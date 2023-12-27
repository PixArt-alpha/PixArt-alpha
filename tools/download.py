# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Functions for downloading pre-trained PixArt models
"""
from torchvision.datasets.utils import download_url
import torch
import os
import argparse


pretrained_models = {'PixArt-XL-2-512x512.pth', 'PixArt-XL-2-1024-MS.pth'}
vae_models = {
    'sd-vae-ft-ema/config.json',
    'sd-vae-ft-ema/diffusion_pytorch_model.bin'
}
t5_models = {
    't5-v1_1-xxl/config.json', 't5-v1_1-xxl/pytorch_model-00001-of-00002.bin',
    't5-v1_1-xxl/pytorch_model-00002-of-00002.bin', 't5-v1_1-xxl/pytorch_model.bin.index.json',
    't5-v1_1-xxl/special_tokens_map.json', 't5-v1_1-xxl/spiece.model',
    't5-v1_1-xxl/tokenizer_config.json',
}


def find_model(model_name):
    """
    Finds a pre-trained G.pt model, downloading it if necessary. Alternatively, loads a model from a local path.
    """
    if model_name in pretrained_models:
        return download_model(model_name)
    assert os.path.isfile(model_name), f'Could not find PixArt checkpoint at {model_name}'
    return torch.load(model_name, map_location=lambda storage, loc: storage)


def download_model(model_name):
    """
    Downloads a pre-trained PixArt model from the web.
    """
    assert model_name in pretrained_models
    local_path = f'output/pretrained_models/{model_name}'
    if not os.path.isfile(local_path):
        os.makedirs('output/pretrained_models', exist_ok=True)
        web_path = f'https://huggingface.co/PixArt-alpha/PixArt-alpha/resolve/main/{model_name}'
        download_url(web_path, 'output/pretrained_models')
    return torch.load(local_path, map_location=lambda storage, loc: storage)


def download_other(model_name, model_zoo, output_dir):
    """
    Downloads a pre-trained PixArt model from the web.
    """
    assert model_name in model_zoo
    local_path = os.path.join(output_dir, model_name)
    if not os.path.isfile(local_path):
        os.makedirs(output_dir, exist_ok=True)
        web_path = f'https://huggingface.co/PixArt-alpha/PixArt-alpha/resolve/main/{model_name}'
        print(web_path)
        download_url(web_path, os.path.join(output_dir, model_name.split('/')[0]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_names', nargs='+', type=str, default=pretrained_models)
    args = parser.parse_args()
    model_names = args.model_names
    model_names = set(model_names)

    # Download PixArt checkpoints
    for t5_model in t5_models:
        download_other(t5_model, t5_models, 'output/pretrained_models/t5_ckpts')
    for vae_model in vae_models:
        download_other(vae_model, vae_models, 'output/pretrained_models/')
    for model in model_names:
        download_model(model)    # for vae_model in vae_models:
    print('Done.')
