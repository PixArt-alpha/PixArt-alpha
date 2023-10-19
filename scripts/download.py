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


pretrained_models = {'PixArt-XL-2-512x512.pt', 'PixArt-XL-2-1024x1024.pt'}


def find_model(model_name):
    """
    Finds a pre-trained G.pt model, downloading it if necessary. Alternatively, loads a model from a local path.
    """
    if model_name in pretrained_models:  # Find/download our pre-trained G.pt checkpoints
        return download_model(model_name)
    else:  # Load a custom PixArt checkpoint:
        assert os.path.isfile(model_name), f'Could not find PixArt checkpoint at {model_name}'
        return torch.load(model_name, map_location=lambda storage, loc: storage)


def download_model(model_name):
    """
    Downloads a pre-trained PixArt model from the web.
    """
    assert model_name in pretrained_models
    local_path = f'pretrained_models/{model_name}'
    if not os.path.isfile(local_path):
        os.makedirs('pretrained_models', exist_ok=True)
        web_path = f'https://huggingface.co/PixArt-alpha/PixArt-alpha/{model_name}'
        download_url(web_path, 'pretrained_models')
    model = torch.load(local_path, map_location=lambda storage, loc: storage)
    return model


if __name__ == "__main__":
    # Download all PixArt checkpoints
    for model in pretrained_models:
        download_model(model)
    print('Done.')
