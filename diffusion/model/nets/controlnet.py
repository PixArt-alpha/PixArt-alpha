from typing import Any, Iterator, Mapping, Union
from torch import Tensor
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from diffusion.model.nets import PixArtMSBlock, PixArtMS
from torch.nn import Module, Linear, init
from copy import deepcopy
from diffusion.model.nets.PixArt import get_2d_sincos_pos_embed
import re
import torch

class ControlPixArtBlock(Module):
    def __init__(self, base_block: PixArtMSBlock) -> None:
        super().__init__()
        self.base_block = base_block
        self.copied_block = deepcopy(base_block)

        for p in self.copied_block.parameters():
            p.requires_grad_(True)

        self.copied_block.load_state_dict(base_block.state_dict())
        self.copied_block.train()
        self.hidden_size = hidden_size = base_block.hidden_size
        self.before_proj = Linear(hidden_size, hidden_size)
        self.after_proj = Linear(hidden_size, hidden_size)
        init.zeros_(self.before_proj.weight)
        init.zeros_(self.before_proj.bias)
        init.zeros_(self.after_proj.weight)
        init.zeros_(self.after_proj.bias)

    def forward(self, x, y, t, mask=None, c=None):
        if c is not None:
            c = self.before_proj(c)
            c = self.copied_block(x + c, y, t, mask)
            c = self.after_proj(c)
        with torch.no_grad():
            x = self.base_block(x, y, t, mask)
        return x + c if c is not None else x

class ControlPixArt(Module):
    def __init__(self, base_model: PixArtMS) -> None:
        super().__init__()
        base_model.eval()
        for p in base_model.parameters():
            p.requires_grad_(False)

        for i in range(len(base_model.blocks)):
            base_model.blocks[i] = ControlPixArtBlock(base_model.blocks[i])

        self.base_model = base_model
    
    # def __getattr__(self, name: str) -> Tensor | Module:
    def __getattr__(self, name: str):
        if name in ['forward', 'forward_with_dpmsolver', 'forward_with_cfg', 'forward_c', 'load_state_dict']:
            return self.__dict__[name]
        elif name == 'base_model':
            return super().__getattr__(name)
        else:
            return getattr(self.base_model, name)

    def forward_c(self, c):
        return self.x_embedder(c) if c is not None else c

    def forward(self, x, t, c, **kwargs):
        return self.base_model(x, t, c=self.forward_c(c), **kwargs)

    def forward_with_dpmsolver(self, x, t, y, data_info, c, **kwargs):
        return self.base_model.forward_with_dpmsolver(x, t, y, data_info=data_info, c=self.forward_c(c), **kwargs)

    def forward_with_cfg(self, x, t, y, cfg_scale, data_info, c, **kwargs):
        return self.base_model.forward_with_cfg(x, t, y, cfg_scale, data_info, c=self.forward_c(c), **kwargs)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        if all(k.startswith('base_model') for k in state_dict.keys()):
            return super().load_state_dict(state_dict, strict)
        else:
            new_key = {}
            for k in state_dict.keys():
                new_key[k] = re.sub(r"(blocks\.\d+)(.*)", r"\1.base_block\2", k)
            for k, v in new_key.items():
                if k != v:
                    print(f"replace {k} to {v}")
                    state_dict[v] = state_dict.pop(k)

            return self.base_model.load_state_dict(state_dict, strict)