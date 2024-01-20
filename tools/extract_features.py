import os
from pathlib import Path
import sys
current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))
from PIL import Image
import torch
from torchvision import transforms as T
import numpy as np
import json
from tqdm import tqdm
import argparse
import threading
from queue import Queue
from pathlib import Path
from torch.utils.data import DataLoader, RandomSampler
from accelerate import Accelerator
from torchvision.transforms.functional import InterpolationMode
from torchvision.datasets.folder import default_loader

from diffusion.model.t5 import T5Embedder
from diffusers.models import AutoencoderKL
from diffusion.data.datasets.InternalData import InternalData
from diffusion.utils.misc import SimpleTimer
from diffusion.utils.data_sampler import AspectRatioBatchSampler
from diffusion.data.builder import DATASETS
from diffusion.data import ASPECT_RATIO_512, ASPECT_RATIO_1024


def get_closest_ratio(height: float, width: float, ratios: dict):
    aspect_ratio = height / width
    closest_ratio = min(ratios.keys(), key=lambda ratio: abs(float(ratio) - aspect_ratio))
    return ratios[closest_ratio], float(closest_ratio)


@DATASETS.register_module()
class DatasetMS(InternalData):
    def __init__(self, root, image_list_json=None, transform=None, resolution=1024, load_vae_feat=False, aspect_ratio_type=None, start_index=0, end_index=100000000, **kwargs):
        if image_list_json is None:
            image_list_json = ['data_info.json']
        assert os.path.isabs(root), 'root must be a absolute path'
        self.root = root
        self.img_dir_name = 'InternalImgs'        # need to change to according to your data structure
        self.json_dir_name = 'InternalData'        # need to change to according to your data structure
        self.transform = transform
        self.load_vae_feat = load_vae_feat
        self.resolution = resolution
        self.meta_data_clean = []
        self.img_samples = []
        self.txt_feat_samples = []
        self.aspect_ratio = aspect_ratio_type
        assert self.aspect_ratio in [ASPECT_RATIO_1024, ASPECT_RATIO_512]
        self.ratio_index = {}
        self.ratio_nums = {}
        for k, v in self.aspect_ratio.items():
            self.ratio_index[float(k)] = []     # used for self.getitem
            self.ratio_nums[float(k)] = 0      # used for batch-sampler

        image_list_json = image_list_json if isinstance(image_list_json, list) else [image_list_json]
        for json_file in image_list_json:
            meta_data = self.load_json(os.path.join(self.root, 'partition', json_file))
            meta_data_clean = [item for item in meta_data if item['ratio'] <= 4]
            self.meta_data_clean.extend(meta_data_clean)
            self.img_samples.extend([os.path.join(self.root.replace(self.json_dir_name, self.img_dir_name), item['path']) for item in meta_data_clean])

        self.img_samples = self.img_samples[start_index: end_index]
        # scan the dataset for ratio static
        for i, info in enumerate(self.meta_data_clean[:len(self.meta_data_clean)//3]):
            ori_h, ori_w = info['height'], info['width']
            closest_size, closest_ratio = get_closest_ratio(ori_h, ori_w, self.aspect_ratio)
            self.ratio_nums[closest_ratio] += 1
            if len(self.ratio_index[closest_ratio]) == 0:
                self.ratio_index[closest_ratio].append(i)

        # Set loader and extensions
        if self.load_vae_feat:
            raise ValueError("No VAE loader here")
        self.loader = default_loader

    def __getitem__(self, idx):
        data_info = {}
        for _ in range(20):
            try:
                img_path = self.img_samples[idx]
                img = self.loader(img_path)
                if self.transform:
                    img = self.transform(img)
                # Calculate closest aspect ratio and resize & crop image[w, h]
                if isinstance(img, Image.Image):
                    h, w = (img.size[1], img.size[0])
                    assert h, w == (self.meta_data_clean[idx]['height'], self.meta_data_clean[idx]['width'])
                    closest_size, closest_ratio = get_closest_ratio(h, w, self.aspect_ratio)
                    closest_size = list(map(lambda x: int(x), closest_size))
                    transform = T.Compose([
                        T.Lambda(lambda img: img.convert('RGB')),
                        T.Resize(closest_size, interpolation=InterpolationMode.BICUBIC),  # Image.BICUBIC
                        T.CenterCrop(closest_size),
                        T.ToTensor(),
                        T.Normalize([.5], [.5]),
                    ])
                    img = transform(img)
                    data_info['img_hw'] = torch.tensor([h, w], dtype=torch.float32)
                    data_info['aspect_ratio'] = closest_ratio
                # change the path according to your data structure
                return img, '_'.join(self.img_samples[idx].rsplit('/', 2)[-2:]) # change from 'serial-number-of-dir/serial-number-of-image.png' ---> 'serial-number-of-dir_serial-number-of-image.png'
            except Exception as e:
                print(f"Error details: {str(e)}")
                idx = np.random.randint(len(self))
        raise RuntimeError('Too many bad data.')

    def get_data_info(self, idx):
        data_info = self.meta_data_clean[idx]
        return {'height': data_info['height'], 'width': data_info['width']}


def extract_caption_t5_do(q):
    while not q.empty():
        item = q.get()
        extract_caption_t5_job(item)
        q.task_done()


def extract_caption_t5_job(item):
    global mutex
    global t5
    global t5_save_dir

    with torch.no_grad():
        caption = item['prompt'].strip()
        if isinstance(caption, str):
            caption = [caption]

        save_path = os.path.join(t5_save_dir, Path(item['path']).stem)
        if os.path.exists(f"{save_path}.npz"):
            return
        try:
            mutex.acquire()
            caption_emb, emb_mask = t5.get_text_embeddings(caption)
            mutex.release()
            emb_dict = {
                'caption_feature': caption_emb.float().cpu().data.numpy(),
                'attention_mask': emb_mask.cpu().data.numpy(),
            }
            np.savez_compressed(save_path, **emb_dict)
        except Exception as e:
            print(e)


def extract_caption_t5():
    global t5
    global t5_save_dir
    # global images_extension
    t5 = T5Embedder(device="cuda", local_cache=True, cache_dir=f'{args.pretrained_models_dir}/t5_ckpts', model_max_length=120)
    t5_save_dir = args.t5_save_root
    os.makedirs(t5_save_dir, exist_ok=True)

    train_data_json = json.load(open(args.json_path, 'r'))
    train_data = train_data_json[args.start_index: args.end_index]

    global mutex
    mutex = threading.Lock()
    jobs = Queue()

    for item in tqdm(train_data):
        jobs.put(item)

    for _ in range(20):
        worker = threading.Thread(target=extract_caption_t5_do, args=(jobs,))
        worker.start()

    jobs.join()


def extract_img_vae_do(q):
    while not q.empty():
        item = q.get()
        extract_img_vae_job(item)
        q.task_done()


def extract_img_vae_job(item):
    return


def extract_img_vae():
    vae = AutoencoderKL.from_pretrained(f'{args.pretrained_models_dir}/sd-vae-ft-ema').to(device)

    train_data_json = json.load(open(args.json_path, 'r'))
    image_names = set()

    vae_save_root = f'{args.vae_save_root}/{image_resize}resolution'
    os.umask(0o000)  # file permission: 666; dir permission: 777
    os.makedirs(vae_save_root, exist_ok=True)

    vae_save_dir = os.path.join(vae_save_root, 'noflip')
    os.makedirs(vae_save_dir, exist_ok=True)

    for item in train_data_json:
        image_name = item['path']
        if image_name in image_names:
            continue
        image_names.add(image_name)
    lines = sorted(image_names)
    lines = lines[args.start_index: args.end_index]

    _, images_extension = os.path.splitext(lines[0])

    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB')),
        T.Resize(image_resize),  # Image.BICUBIC
        T.CenterCrop(image_resize),
        T.ToTensor(),
        T.Normalize([.5], [.5]),
    ])

    os.umask(0o000)  # file permission: 666; dir permission: 777
    for image_name in tqdm(lines):
        save_path = os.path.join(vae_save_dir, Path(image_name).stem)
        if os.path.exists(f"{save_path}.npy"):
            continue
        try:
            img = Image.open(f'{args.dataset_root}/{image_name}')
            img = transform(img).to(device)[None]

            with torch.no_grad():
                posterior = vae.encode(img).latent_dist
                z = torch.cat([posterior.mean, posterior.std], dim=1).detach().cpu().numpy().squeeze()

            np.save(save_path, z)
        except Exception as e:
            print(e)
            print(image_name)


def save_results(results, paths, signature, work_dir):
    timer = SimpleTimer(len(results), log_interval=100, desc="Saving Results")
    # save to npy
    new_paths = []
    os.umask(0o000)  # file permission: 666; dir permission: 777
    for res, p in zip(results, paths):
        file_name = p.split('.')[0] + '.npy'
        new_folder = signature
        save_folder = os.path.join(work_dir, new_folder)
        if os.path.exists(save_folder):
            raise FileExistsError(f"{save_folder} exists. BE careful not to overwrite your files. Comment this error raising for overwriting!!")
        os.makedirs(save_folder, exist_ok=True)
        new_paths.append(os.path.join(new_folder, file_name))
        np.save(os.path.join(save_folder, file_name), res)
        timer.log()
    # save paths
    with open(os.path.join(work_dir, f"VAE-{signature}.txt"), 'w') as f:
        f.write('\n'.join(new_paths))


def inference(vae, dataloader, signature, work_dir):
    timer = SimpleTimer(len(dataloader), log_interval=100, desc="VAE-Inference")

    for batch in dataloader:
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True):
                posterior = vae.encode(batch[0]).latent_dist
                results = torch.cat([posterior.mean, posterior.std], dim=1).detach().cpu().numpy()
        path = batch[1]
        save_results(results, path, signature=signature, work_dir=work_dir)
        timer.log()


def extract_img_vae_multiscale(bs=1):

    assert image_resize in [512, 1024]
    work_dir = os.path.abspath(args.vae_save_root)
    os.umask(0o000)  # file permission: 666; dir permission: 777
    os.makedirs(work_dir, exist_ok=True)
    accelerator = Accelerator(mixed_precision='fp16')
    vae = AutoencoderKL.from_pretrained(f'{args.pretrained_models_dir}/sd-vae-ft-ema').to(device)

    signature = 'ms'

    aspect_ratio_type = ASPECT_RATIO_1024 if image_resize == 1024 else ASPECT_RATIO_512
    dataset = DatasetMS(args.dataset_root, image_list_json=[args.json_file], transform=None, sample_subset=None,
                        aspect_ratio_type=aspect_ratio_type, start_index=args.start_index, end_index=args.end_index)

    # create AspectRatioBatchSampler
    sampler = AspectRatioBatchSampler(sampler=RandomSampler(dataset), dataset=dataset, batch_size=bs, aspect_ratios=dataset.aspect_ratio, ratio_nums=dataset.ratio_nums)

    # create DataLoader
    dataloader = DataLoader(dataset, batch_sampler=sampler, num_workers=13, pin_memory=True)
    dataloader = accelerator.prepare(dataloader, )

    inference(vae, dataloader, signature=signature, work_dir=work_dir)
    accelerator.wait_for_everyone()

    print('done')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--multi_scale", action='store_true', default=False, help="multi-scale feature extraction")
    parser.add_argument("--img_size", default=512, type=int, help="image scale for multi-scale feature extraction")
    parser.add_argument('--start_index', default=0, type=int)
    parser.add_argument('--end_index', default=1000000, type=int)
    
    parser.add_argument('--json_path', type=str)
    parser.add_argument('--t5_save_root', default='data/data_toy/caption_feature_wmask', type=str)
    parser.add_argument('--vae_save_root', default='data/data_toy/img_vae_features', type=str)
    parser.add_argument('--dataset_root', default='data/data_toy', type=str)
    parser.add_argument('--pretrained_models_dir', default='output/pretrained_models', type=str)

    ### for multi-scale(ms) vae feauture extraction
    parser.add_argument('--json_file', type=str)
    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_resize = args.img_size

    # prepare extracted caption t5 features for training
    extract_caption_t5()

    # prepare extracted image vae features for training
    if args.multi_scale:
        print(f'Extracting Multi-scale Image Resolution based on {image_resize}')
        extract_img_vae_multiscale(bs=1)    # recommend bs = 1 for AspectRatioBatchSampler
    else:
        print(f'Extracting Single Image Resolution {image_resize}')
        extract_img_vae()