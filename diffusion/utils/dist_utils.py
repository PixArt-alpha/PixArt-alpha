"""
This file contains primitives for multi-gpu communication.
This is useful when doing distributed training.
"""
import os
import pickle
import shutil

import gc
import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info


def is_distributed():
    return get_world_size() > 1


def get_world_size():
    if not dist.is_available():
        return 1
    return dist.get_world_size() if dist.is_initialized() else 1


def get_rank():
    if not dist.is_available():
        return 0
    return dist.get_rank() if dist.is_initialized() else 0


def get_local_rank():
    if not dist.is_available():
        return 0
    return int(os.getenv('LOCAL_RANK', 0)) if dist.is_initialized() else 0


def is_master():
    return get_rank() == 0


def is_local_master():
    return get_local_rank() == 0


def get_local_proc_group(group_size=8):
    world_size = get_world_size()
    if world_size <= group_size or group_size == 1:
        return None
    assert world_size % group_size == 0, f'world size ({world_size}) should be evenly divided by group size ({group_size}).'
    process_groups = getattr(get_local_proc_group, 'process_groups', {})
    if group_size not in process_groups:
        num_groups = dist.get_world_size() // group_size
        groups = [list(range(i * group_size, (i + 1) * group_size)) for i in range(num_groups)]
        process_groups.update({group_size: [torch.distributed.new_group(group) for group in groups]})
        get_local_proc_group.process_groups = process_groups

    group_idx = get_rank() // group_size
    return get_local_proc_group.process_groups.get(group_size)[group_idx]


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    to_device = torch.device("cuda")
    # to_device = torch.device("cpu")

    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(to_device)

    # obtain Tensor size of each rank
    local_size = torch.LongTensor([tensor.numel()]).to(to_device)
    size_list = [torch.LongTensor([0]).to(to_device) for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    tensor_list = [
        torch.ByteTensor(size=(max_size,)).to(to_device) for _ in size_list
    ]
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size,)).to(to_device)
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        reduced_dict = _extracted_from_reduce_dict_14(input_dict, average, world_size)
    return reduced_dict


# TODO Rename this here and in `reduce_dict`
def _extracted_from_reduce_dict_14(input_dict, average, world_size):
    names = []
    values = []
    # sort the keys so that they are consistent across processes
    for k in sorted(input_dict.keys()):
        names.append(k)
        values.append(input_dict[k])
    values = torch.stack(values, dim=0)
    dist.reduce(values, dst=0)
    if dist.get_rank() == 0 and average:
        # only main process gets accumulated, so only divide by
        # world_size in this case
        values /= world_size
    return dict(zip(names, values))


def broadcast(data, **kwargs):
    if get_world_size() == 1:
        return data
    data = [data]
    dist.broadcast_object_list(data, **kwargs)
    return data[0]


def all_gather_cpu(result_part, tmpdir=None, collect_by_master=True):
    rank, world_size = get_dist_info()
    if tmpdir is None:
        tmpdir = './tmp'
    if rank == 0:
        mmcv.mkdir_or_exist(tmpdir)
    synchronize()
    # dump the part result to the dir
    mmcv.dump(result_part, os.path.join(tmpdir, f'part_{rank}.pkl'))
    synchronize()
    if collect_by_master and rank != 0:
        return None
    # load results of all parts from tmp dir
    results = []
    for i in range(world_size):
        part_file = os.path.join(tmpdir, f'part_{i}.pkl')
        results.append(mmcv.load(part_file))
    if not collect_by_master:
        synchronize()
    # remove tmp dir
    if rank == 0:
        shutil.rmtree(tmpdir)
    return results

def all_gather_tensor(tensor, group_size=None, group=None):
    if group_size is None:
        group_size = get_world_size()
    if group_size == 1:
        output = [tensor]
    else:
        output = [torch.zeros_like(tensor) for _ in range(group_size)]
        dist.all_gather(output, tensor, group=group)
    return output


def gather_difflen_tensor(feat, num_samples_list, concat=True, group=None, group_size=None):
    world_size = get_world_size()
    if world_size == 1:
        return feat if concat else [feat]
    num_samples, *feat_dim = feat.size()
    # padding to max number of samples
    feat_padding = feat.new_zeros((max(num_samples_list), *feat_dim))
    feat_padding[:num_samples] = feat
    # gather
    feat_gather = all_gather_tensor(feat_padding, group=group, group_size=group_size)
    for r, num in enumerate(num_samples_list):
        feat_gather[r] = feat_gather[r][:num]
    if concat:
        feat_gather = torch.cat(feat_gather)
    return feat_gather


class GatherLayer(torch.autograd.Function):
    '''Gather tensors from all process, supporting backward propagation.
    '''

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        num_samples = torch.tensor(input.size(0), dtype=torch.long, device=input.device)
        ctx.num_samples_list = all_gather_tensor(num_samples)
        output = gather_difflen_tensor(input, ctx.num_samples_list, concat=False)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):  # tuple(output)'s grad
        input, = ctx.saved_tensors
        num_samples_list = ctx.num_samples_list
        rank = get_rank()
        start, end = sum(num_samples_list[:rank]), sum(num_samples_list[:rank + 1])
        grads = torch.cat(grads)
        if is_distributed():
            dist.all_reduce(grads)
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[start:end]
        return grad_out, None, None


class GatherLayerWithGroup(torch.autograd.Function):
    '''Gather tensors from all process, supporting backward propagation.
    '''

    @staticmethod
    def forward(ctx, input, group, group_size):
        ctx.save_for_backward(input)
        ctx.group_size = group_size
        output = all_gather_tensor(input, group=group, group_size=group_size)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):  # tuple(output)'s grad
        input, = ctx.saved_tensors
        grads = torch.stack(grads)
        if is_distributed():
            dist.all_reduce(grads)
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[get_rank() % ctx.group_size]
        return grad_out, None, None


def gather_layer_with_group(data, group=None, group_size=None):
    if group_size is None:
        group_size = get_world_size()
    return GatherLayer.apply(data, group, group_size)

from typing import Union
import math
# from torch.distributed.fsdp.fully_sharded_data_parallel import TrainingState_, _calc_grad_norm

@torch.no_grad()
def clip_grad_norm_(
    self, max_norm: Union[float, int], norm_type: Union[float, int] = 2.0
) -> None:
    self._lazy_init()
    self._wait_for_previous_optim_step()
    assert self._is_root, "clip_grad_norm should only be called on the root (parent) instance"
    self._assert_state(TrainingState_.IDLE)

    max_norm = float(max_norm)
    norm_type = float(norm_type)
    # Computes the max norm for this shard's gradients and sync's across workers
    local_norm = _calc_grad_norm(self.params_with_grad, norm_type).cuda()  # type: ignore[arg-type]
    if norm_type == math.inf:
        total_norm = local_norm
        dist.all_reduce(total_norm, op=torch.distributed.ReduceOp.MAX, group=self.process_group)
    else:
        total_norm = local_norm ** norm_type
        dist.all_reduce(total_norm, group=self.process_group)
        total_norm = total_norm ** (1.0 / norm_type)

    clip_coef = torch.tensor(max_norm, dtype=total_norm.dtype, device=total_norm.device) / (total_norm + 1e-6)
    if clip_coef < 1:
        # multiply by clip_coef, aka, (max_norm/total_norm).
        for p in self.params_with_grad:
            assert p.grad is not None
            p.grad.detach().mul_(clip_coef.to(p.grad.device))
    return total_norm


def flush():
    gc.collect()
    torch.cuda.empty_cache()
