# {'model': 'LLaVA-7B-v0', 'prompt': 'You are LLaVA, a large language and vision assistant trained by UW Madison WAIV Lab.You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.Follow the instructions carefully and explain your answers in detail.###Human: Hi!###Assistant: Hi there!  How can I help you today?\n###Human: ?\n<image>###Assistant:', 'temperature': 0.2, 'max_new_tokens': 512, 'stop': '###', 'images': "List of 1 images: ['793f00027d3dc5bd69445a388a2f289c']"}
import sys
from pathlib import Path
current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))
import argparse
import torch
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, AutoConfig
from diffusion.model.llava import LlavaMPTForCausalLM
from PIL import Image
from tqdm import tqdm
from os import path, makedirs
from torch.utils.data import Dataset, DataLoader
import json


DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


def expand2square(pil_img, background_color=(122, 116, 104)):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def pad2square(image):
    max_hw, min_hw = max(image.size), min(image.size)
    aspect_ratio = max_hw / min_hw
    max_len, min_len = 800, 400
    shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
    longest_edge = int(shortest_edge * aspect_ratio)
    W, H = image.size
    if H > W:
        H, W = longest_edge, shortest_edge
    else:
        H, W = shortest_edge, longest_edge
    image = image.resize((W, H))
    return image


def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = LlavaMPTForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

    vision_tower = model.get_model().vision_tower[0]
    if vision_tower.device.type == 'meta':
        vision_tower = CLIPVisionModel.from_pretrained(
            vision_tower.config._name_or_path, torch_dtype=torch.float16, low_cpu_mem_usage=True).cuda()
        model.get_model().vision_tower[0] = vision_tower
    else:
        vision_tower.to(device='cuda', dtype=torch.float16)
    vision_config = vision_tower.config
    vision_config.im_patch_token = tokenizer.convert_tokens_to_ids(
        [DEFAULT_IMAGE_PATCH_TOKEN])[0]
    vision_config.use_im_start_end = mm_use_im_start_end
    if mm_use_im_start_end:
        vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])

    model.cuda()

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, context_len


class SanitizedLaion(Dataset):
    def __init__(self, root_dir, index_file, prompt, config, img_extension='.jpg', caption=True) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.image_processor = CLIPImageProcessor.from_pretrained(AutoConfig.from_pretrained(config).mm_vision_tower, torch_dtype=torch.float16)
        self.prompt = prompt
        self.img_extension = img_extension
        self.caption=caption

        if '.txt' in index_file:
            with open(index_file, 'r') as f:
                self.lines = f.readlines()
        elif '.json' in index_file:
            with open(index_file, 'r') as f:
                self.lines = json.load(f)
        else:
            raise ValueError(f'{index_file} format not supported')

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        item = self.lines[idx]
        caption = item['prompt'].strip()
        prompt = self.prompt.format(caption) if self.caption else self.prompt
        with open(path.join(self.root_dir, item['path']), 'rb') as f:
            img = pad2square(Image.open(f).convert('RGB'))
        return self.image_processor(img, return_tensors='pt')['pixel_values'].squeeze(), prompt, item['path'].split(self.img_extension)[0]


@torch.no_grad()
def caption(tokenizer, model, context_len, images, prompt, prefix):
    images = images.to(model.device, dtype=torch.float16)
    # HACK: 256 is the max image token length hacked
    replace_token = DEFAULT_IMAGE_PATCH_TOKEN * 256
    if getattr(model.config, 'mm_use_im_start_end', False):
        replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN

    prompt = list(map(lambda p: p.replace(DEFAULT_IMAGE_TOKEN, replace_token), prompt))

    temperature = 0.2
    max_new_tokens = 1024
    stop_str = '<|im_end|>'

    max_src_len = context_len - max_new_tokens - 8
    input_ids = tokenizer(prompt).input_ids
    input_ids = list(map(lambda input_id: input_id[-max_src_len:], input_ids))
    lens = list(map(lambda x: len(x), input_ids))
    longest = max(lens)
    input_ids = list(map(lambda x: x if len(x) == longest else [tokenizer.pad_token_id] * (longest - len(x)) + x, input_ids))

    pred_ids = torch.zeros([images.shape[0], 0], device=model.device, dtype=torch.long)
    past_key_values = None
    finish = [False] * images.shape[0]
    for i in tqdm(range(max_new_tokens), leave=False):
        if i == 0:
            out = model(
                torch.as_tensor(input_ids).cuda(),
                use_cache=True,
                images=images)
            del images
        else:
            attention_mask = torch.ones(1, past_key_values[0][0].shape[-2] + 1, device="cuda")
            out = model(input_ids=token,
                        use_cache=True,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values)
        past_key_values = out.past_key_values
        logits = out.logits
        last_token_logits = logits[:, -1]
        if temperature < 1e-4:
            token = torch.argmax(last_token_logits)
        else:
            probs = torch.softmax(last_token_logits / temperature, dim=-1)
            token = torch.multinomial(probs, num_samples=1)

        pred_ids = torch.concatenate([pred_ids, token], dim=1)

        for ii in torch.nonzero(token.cpu() == tokenizer.eos_token_id, as_tuple=True)[0]:
            if finish[ii]:
                continue
            ii = int(ii)
            output = tokenizer.decode(pred_ids[ii][:-1]).removesuffix(stop_str)
            finish[ii] = True
            yield output, prefix[ii]

        if all(finish):
            break



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/LLaVA-Lightning-MPT-7B-preview")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument('--index', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    prompt = """<|im_start|>system
    - You are LLaVA, a large language and vision assistant trained by UW Madison WAIV Lab.
    - You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.
    - You should follow the instructions carefully and explain your answers in detail.<|im_end|><|im_start|>user
    Given the caption of this image "{}", describe this image in a very detailed manner
    <image><|im_end|><|im_start|>assistant\n"""

    prompt_nocap = """<|im_start|>system
    - You are LLaVA, a large language and vision assistant trained by UW Madison WAIV Lab.
    - You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.
    - You should follow the instructions carefully and explain your answers in detail.<|im_end|><|im_start|>user
    Describe this image in a very detailed manner
    <image><|im_end|><|im_start|>assistant\n"""
    d = SanitizedLaion(args.data_root, args.index, prompt, args.model_path, img_extension='.png')
    l = DataLoader(d, batch_size=32, pin_memory=True, num_workers=10)

    tokenizer, model, context_len = load_model(args.model_path)
    # model = torch.compile(model)
    for b in tqdm(l):
        for c, p in caption(tokenizer, model, context_len, *b):
            o = path.join(args.output, f'{p}.txt')
            makedirs(path.dirname(o), exist_ok=True, mode=0o755)
            with open(o, 'w') as k:
                k.write(c)
