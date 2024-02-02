# -*- coding: utf-8 -*-
import os
import re
import html
import urllib.parse as ul

import ftfy
import torch
from bs4 import BeautifulSoup
from transformers import T5EncoderModel, AutoTokenizer
from huggingface_hub import hf_hub_download

class T5Embedder:

    available_models = ['t5-v1_1-xxl']
    bad_punct_regex = re.compile(r'['+'#®•©™&@·º½¾¿¡§~'+'\)'+'\('+'\]'+'\['+'\}'+'\{'+'\|'+'\\'+'\/'+'\*' + r']{1,}')  # noqa

    def __init__(self, device, dir_or_name='t5-v1_1-xxl', *, local_cache=False, cache_dir=None, hf_token=None, use_text_preprocessing=True,
                 t5_model_kwargs=None, torch_dtype=None, use_offload_folder=None, model_max_length=120):
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype or torch.bfloat16
        if t5_model_kwargs is None:
            t5_model_kwargs = {'low_cpu_mem_usage': True, 'torch_dtype': self.torch_dtype}
            if use_offload_folder is not None:
                t5_model_kwargs['offload_folder'] = use_offload_folder
                t5_model_kwargs['device_map'] = {
                    'shared': self.device,
                    'encoder.embed_tokens': self.device,
                    'encoder.block.0': self.device,
                    'encoder.block.1': self.device,
                    'encoder.block.2': self.device,
                    'encoder.block.3': self.device,
                    'encoder.block.4': self.device,
                    'encoder.block.5': self.device,
                    'encoder.block.6': self.device,
                    'encoder.block.7': self.device,
                    'encoder.block.8': self.device,
                    'encoder.block.9': self.device,
                    'encoder.block.10': self.device,
                    'encoder.block.11': self.device,
                    'encoder.block.12': 'disk',
                    'encoder.block.13': 'disk',
                    'encoder.block.14': 'disk',
                    'encoder.block.15': 'disk',
                    'encoder.block.16': 'disk',
                    'encoder.block.17': 'disk',
                    'encoder.block.18': 'disk',
                    'encoder.block.19': 'disk',
                    'encoder.block.20': 'disk',
                    'encoder.block.21': 'disk',
                    'encoder.block.22': 'disk',
                    'encoder.block.23': 'disk',
                    'encoder.final_layer_norm': 'disk',
                    'encoder.dropout': 'disk',
                }
            else:
                t5_model_kwargs['device_map'] = {'shared': self.device, 'encoder': self.device}

        self.use_text_preprocessing = use_text_preprocessing
        self.hf_token = hf_token
        self.cache_dir = cache_dir or os.path.expanduser('~/.cache/IF_')
        self.dir_or_name = dir_or_name
        tokenizer_path, path = dir_or_name, dir_or_name
        if local_cache:
            cache_dir = os.path.join(self.cache_dir, dir_or_name)
            tokenizer_path, path = cache_dir, cache_dir
        elif dir_or_name in self.available_models:
            cache_dir = os.path.join(self.cache_dir, dir_or_name)
            for filename in [
                'config.json', 'special_tokens_map.json', 'spiece.model', 'tokenizer_config.json',
                'pytorch_model.bin.index.json', 'pytorch_model-00001-of-00002.bin', 'pytorch_model-00002-of-00002.bin'
            ]:
                hf_hub_download(repo_id=f'DeepFloyd/{dir_or_name}', filename=filename, cache_dir=cache_dir,
                                force_filename=filename, token=self.hf_token)
            tokenizer_path, path = cache_dir, cache_dir
        else:
            cache_dir = os.path.join(self.cache_dir, 't5-v1_1-xxl')
            for filename in [
                'config.json', 'special_tokens_map.json', 'spiece.model', 'tokenizer_config.json',
            ]:
                hf_hub_download(repo_id='DeepFloyd/t5-v1_1-xxl', filename=filename, cache_dir=cache_dir,
                                force_filename=filename, token=self.hf_token)
            tokenizer_path = cache_dir

        print(tokenizer_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = T5EncoderModel.from_pretrained(path, **t5_model_kwargs).eval()
        self.model_max_length = model_max_length

    def get_text_embeddings(self, texts):
        texts = [self.text_preprocessing(text) for text in texts]

        text_tokens_and_mask = self.tokenizer(
            texts,
            max_length=self.model_max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )

        text_tokens_and_mask['input_ids'] = text_tokens_and_mask['input_ids']
        text_tokens_and_mask['attention_mask'] = text_tokens_and_mask['attention_mask']

        with torch.no_grad():
            text_encoder_embs = self.model(
                input_ids=text_tokens_and_mask['input_ids'].to(self.device),
                attention_mask=text_tokens_and_mask['attention_mask'].to(self.device),
            )['last_hidden_state'].detach()
        return text_encoder_embs, text_tokens_and_mask['attention_mask'].to(self.device)

    def text_preprocessing(self, text):
        if self.use_text_preprocessing:
            # The exact text cleaning as was in the training stage:
            text = self.clean_caption(text)
            text = self.clean_caption(text)
            return text
        else:
            return text.lower().strip()

    @staticmethod
    def basic_clean(text):
        text = ftfy.fix_text(text)
        text = html.unescape(html.unescape(text))
        return text.strip()

    def clean_caption(self, caption):
        caption = str(caption)
        caption = ul.unquote_plus(caption)
        caption = caption.strip().lower()
        caption = str.replace('<person>', 'person', caption)
        # urls:
        caption = str.replace(
            r'\b((?:https?:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))',  # noqa
            '', caption)  # regex for urls
        caption = str.replace(
            r'\b((?:www:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))',  # noqa
            '', caption)  # regex for urls
        # html:
        caption = BeautifulSoup(caption, features='html.parser').text

        # @<nickname>
        caption = str.replace(r'@[\w\d]+\b', '', caption)

        # 31C0—31EF CJK Strokes
        # 31F0—31FF Katakana Phonetic Extensions
        # 3200—32FF Enclosed CJK Letters and Months
        # 3300—33FF CJK Compatibility
        # 3400—4DBF CJK Unified Ideographs Extension A
        # 4DC0—4DFF Yijing Hexagram Symbols
        # 4E00—9FFF CJK Unified Ideographs
        caption = str.replace(r'[\u31c0-\u31ef]+', '', caption)
        caption = str.replace(r'[\u31f0-\u31ff]+', '', caption)
        caption = str.replace(r'[\u3200-\u32ff]+', '', caption)
        caption = str.replace(r'[\u3300-\u33ff]+', '', caption)
        caption = str.replace(r'[\u3400-\u4dbf]+', '', caption)
        caption = str.replace(r'[\u4dc0-\u4dff]+', '', caption)
        caption = str.replace(r'[\u4e00-\u9fff]+', '', caption)
        #######################################################

        # все виды тире / all types of dash --> "-"
        caption = str.replace(
            r'[\u002D\u058A\u05BE\u1400\u1806\u2010-\u2015\u2E17\u2E1A\u2E3A\u2E3B\u2E40\u301C\u3030\u30A0\uFE31\uFE32\uFE58\uFE63\uFF0D]+',  # noqa
            '-', caption)

        # кавычки к одному стандарту
        caption = str.replace(r'[`´«»“”¨]', '"', caption)
        caption = str.replace(r'[‘’]', "'", caption)

        # &quot;
        caption = str.replace(r'&quot;?', '', caption)
        # &amp
        caption = str.replace(r'&amp', '', caption)

        # ip adresses:
        caption = str.replace(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', ' ', caption)

        # article ids:
        caption = str.replace(r'\d:\d\d\s+$', '', caption)

        # \n
        caption = str.replace(r'\\n', ' ', caption)

        # "#123"
        caption = str.replace(r'#\d{1,3}\b', '', caption)
        # "#12345.."
        caption = str.replace(r'#\d{5,}\b', '', caption)
        # "123456.."
        caption = str.replace(r'\b\d{6,}\b', '', caption)
        # filenames:
        caption = str.replace(r'[\S]+\.(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)', '', caption)

        #
        caption = str.replace(r'[\"\']{2,}', r'"', caption)  # """AUSVERKAUFT"""
        caption = str.replace(r'[\.]{2,}', r' ', caption)  # """AUSVERKAUFT"""

        caption = str.replace(self.bad_punct_regex, r' ', caption)  # ***AUSVERKAUFT***, #AUSVERKAUFT
        caption = str.replace(r'\s+\.\s+', r' ', caption)  # " . "

        # this-is-my-cute-cat / this_is_my_cute_cat
        regex2 = re.compile(r'(?:\-|\_)')
        if len(re.findall(regex2, caption)) > 3:
            caption = str.replace(regex2, ' ', caption)

        caption = self.basic_clean(caption)

        caption = str.replace(r'\b[a-zA-Z]{1,3}\d{3,15}\b', '', caption)  # jc6640
        caption = str.replace(r'\b[a-zA-Z]+\d+[a-zA-Z]+\b', '', caption)  # jc6640vc
        caption = str.replace(r'\b\d+[a-zA-Z]+\d+\b', '', caption)  # 6640vc231

        caption = str.replace(r'(worldwide\s+)?(free\s+)?shipping', '', caption)
        caption = str.replace(r'(free\s)?download(\sfree)?', '', caption)
        caption = str.replace(r'\bclick\b\s(?:for|on)\s\w+', '', caption)
        caption = str.replace(r'\b(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)(\simage[s]?)?', '', caption)
        caption = str.replace(r'\bpage\s+\d+\b', '', caption)

        caption = str.replace(r'\b\d*[a-zA-Z]+\d+[a-zA-Z]+\d+[a-zA-Z\d]*\b', r' ', caption)  # j2d1a2a...

        caption = str.replace(r'\b\d+\.?\d*[xх×]\d+\.?\d*\b', '', caption)

        caption = str.replace(r'\b\s+\:\s+', r': ', caption)
        caption = str.replace(r'(\D[,\./])\b', r'\1 ', caption)
        caption = str.replace(r'\s+', ' ', caption)

        caption.strip()

        caption = str.replace(r'^[\"\']([\w\W]+)[\"\']$', r'\1', caption)
        caption = str.replace(r'^[\'\_,\-\:;]', r'', caption)
        caption = str.replace(r'[\'\_,\-\:\-\+]$', r'', caption)
        caption = str.replace(r'^\.\S+$', '', caption)

        return caption.strip()
