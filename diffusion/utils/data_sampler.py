# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Sequence
from torch.utils.data import BatchSampler, Sampler, Dataset
from random import shuffle, choice
from copy import deepcopy
from diffusion.utils.logger import get_root_logger


class AspectRatioBatchSampler(BatchSampler):
    """A sampler wrapper for grouping images with similar aspect ratio into a same batch.

    Args:
        sampler (Sampler): Base sampler.
        dataset (Dataset): Dataset providing data information.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``.
        aspect_ratios (dict): The predefined aspect ratios.
    """

    def __init__(self,
                 sampler: Sampler,
                 dataset: Dataset,
                 batch_size: int,
                 aspect_ratios: dict,
                 drop_last: bool = False,
                 config=None,
                 valid_num=0,   # take as valid aspect-ratio when sample number >= valid_num
                 **kwargs) -> None:
        if not isinstance(sampler, Sampler):
            raise TypeError('sampler should be an instance of ``Sampler``, '
                            f'but got {sampler}')
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError('batch_size should be a positive integer value, '
                             f'but got batch_size={batch_size}')
        self.sampler = sampler
        self.dataset = dataset
        self.batch_size = batch_size
        self.aspect_ratios = aspect_ratios
        self.drop_last = drop_last
        self.ratio_nums_gt = kwargs.get('ratio_nums', None)
        self.config = config
        assert self.ratio_nums_gt
        # buckets for each aspect ratio
        self._aspect_ratio_buckets = {ratio: [] for ratio in aspect_ratios.keys()}
        self.current_available_bucket_keys =  [str(k) for k, v in self.ratio_nums_gt.items() if v >= valid_num]
        logger = get_root_logger() if config is None else get_root_logger(os.path.join(config.work_dir, 'train_log.log'))
        logger.warning(f"Using valid_num={valid_num} in config file. Available {len(self.current_available_bucket_keys)} aspect_ratios: {self.current_available_bucket_keys}")

    def __iter__(self) -> Sequence[int]:
        for idx in self.sampler:
            data_info = self.dataset.get_data_info(idx)
            height, width =  data_info['height'], data_info['width']
            ratio = height / width
            # find the closest aspect ratio
            closest_ratio = min(self.aspect_ratios.keys(), key=lambda r: abs(float(r) - ratio))
            if closest_ratio not in self.current_available_bucket_keys:
                continue
            bucket = self._aspect_ratio_buckets[closest_ratio]
            bucket.append(idx)
            # yield a batch of indices in the same aspect ratio group
            if len(bucket) == self.batch_size:
                yield bucket[:]
                del bucket[:]

        # yield the rest data and reset the buckets
        for bucket in self._aspect_ratio_buckets.values():
            while len(bucket) > 0:
                if len(bucket) <= self.batch_size:
                    if not self.drop_last:
                        yield bucket[:]
                    bucket = []
                else:
                    yield bucket[:self.batch_size]
                    bucket = bucket[self.batch_size:]


class BalancedAspectRatioBatchSampler(AspectRatioBatchSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Assign samples to each bucket
        self.ratio_nums_gt = kwargs.get('ratio_nums', None)
        assert self.ratio_nums_gt
        self._aspect_ratio_buckets = {float(ratio): [] for ratio in self.aspect_ratios.keys()}
        self.original_buckets = {}
        self.current_available_bucket_keys =  [k for k, v in self.ratio_nums_gt.items() if v >= 3000]
        self.all_available_keys = deepcopy(self.current_available_bucket_keys)
        self.exhausted_bucket_keys = []
        self.total_batches = len(self.sampler) // self.batch_size
        self._aspect_ratio_count = {}
        for k in self.all_available_keys:
            self._aspect_ratio_count[float(k)] = 0
            self.original_buckets[float(k)] = []
        logger = get_root_logger(os.path.join(self.config.work_dir, 'train_log.log'))
        logger.warning(f"Available {len(self.current_available_bucket_keys)} aspect_ratios: {self.current_available_bucket_keys}")

    def __iter__(self) -> Sequence[int]:
        i = 0
        for idx in self.sampler:
            data_info = self.dataset.get_data_info(idx)
            height, width = data_info['height'], data_info['width']
            ratio = height / width
            closest_ratio = float(min(self.aspect_ratios.keys(), key=lambda r: abs(float(r) - ratio)))
            if closest_ratio not in self.all_available_keys:
                continue
            if self._aspect_ratio_count[closest_ratio] < self.ratio_nums_gt[closest_ratio]:
                self._aspect_ratio_count[closest_ratio] += 1
                self._aspect_ratio_buckets[closest_ratio].append(idx)
                self.original_buckets[closest_ratio].append(idx)    # Save the original samples for each bucket
            if not self.current_available_bucket_keys:
                self.current_available_bucket_keys, self.exhausted_bucket_keys = self.exhausted_bucket_keys, []

            if closest_ratio not in self.current_available_bucket_keys:
                continue
            key = closest_ratio
            bucket = self._aspect_ratio_buckets[key]
            if len(bucket) == self.batch_size:
                yield bucket[:self.batch_size]
                del bucket[:self.batch_size]
                i += 1
                self.exhausted_bucket_keys.append(key)
                self.current_available_bucket_keys.remove(key)

        for _ in range(self.total_batches - i):
            key = choice(self.all_available_keys)
            bucket = self._aspect_ratio_buckets[key]
            if len(bucket) >= self.batch_size:
                yield bucket[:self.batch_size]
                del bucket[:self.batch_size]

                # If a bucket is exhausted
                if not bucket:
                    self._aspect_ratio_buckets[key] = deepcopy(self.original_buckets[key][:])
                    shuffle(self._aspect_ratio_buckets[key])
            else:
                self._aspect_ratio_buckets[key] = deepcopy(self.original_buckets[key][:])
                shuffle(self._aspect_ratio_buckets[key])
