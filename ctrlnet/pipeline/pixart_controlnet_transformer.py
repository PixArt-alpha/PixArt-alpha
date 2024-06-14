from typing import Any, Dict, Optional

import torch
from torch import nn

from diffusers.models.attention import BasicTransformerBlock
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

class PixArtControlNetAdapterBlock(nn.Module):
    def __init__(
        self, 
        block_index,
        
        # taken from PixArtTransformer2DModel
        num_attention_heads: int = 16,
        attention_head_dim: int = 72,
        dropout: float = 0.0,
        cross_attention_dim: Optional[int] = 1152,
        attention_bias: bool = True,
        activation_fn: str = "gelu-approximate",
        num_embeds_ada_norm: Optional[int] = 1000,
        upcast_attention: bool = False,
        norm_type: str = "ada_norm_single",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-6,
        attention_type: Optional[str] = "default",
    ):
        super().__init__()

        self.block_index = block_index
        self.inner_dim = num_attention_heads * attention_head_dim

        # the first block has a zero before layer
        if self.block_index == 0:
            self.before_proj = nn.Linear(self.inner_dim, self.inner_dim)
            nn.init.zeros_(self.before_proj.weight)
            nn.init.zeros_(self.before_proj.bias)

        self.transformer_block = BasicTransformerBlock(
            self.inner_dim,
            num_attention_heads,
            attention_head_dim,
            dropout=dropout,
            cross_attention_dim=cross_attention_dim,
            activation_fn=activation_fn,
            num_embeds_ada_norm=num_embeds_ada_norm,
            attention_bias=attention_bias,
            upcast_attention=upcast_attention,
            norm_type=norm_type,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            attention_type=attention_type,
        )

        self.after_proj = nn.Linear(self.inner_dim, self.inner_dim) 
        nn.init.zeros_(self.after_proj.weight)
        nn.init.zeros_(self.after_proj.bias)

class PixArtControlNetAdapterModel(ModelMixin, ConfigMixin):
    # N=13, as specified in the paper https://arxiv.org/html/2401.05252v1/#S4 ControlNet-Transformer
    def __init__(self, num_layers = 13) -> None:
        super().__init__()

        self.controlnet_blocks = nn.ModuleList(
            [
                PixArtControlNetAdapterBlock(block_index=i)
                for i in range(num_layers)
            ]
        )

class PixArtControlNetTransformerModel(nn.Module):
    def __init__(self, *args, blocks_num=13, **kwargs):
        super().__init__(*args, **kwargs)

        self.blocks_num = blocks_num
        self.transformer_blocks = nn.ModuleList()
    
    def forward(self, x, timestep, y, mask=None, data_info=None, c=None, **kwargs):
        if c is not None:
            c = c.to(self.dtype)
            c = self.forward_c(c)

        x = x.to(self.dtype)
        timestep = timestep.to(self.dtype)
        y = y.to(self.dtype)

        x = self.pos_embed(x)
        t = self.adaln_single.emb.timestep_embedder(timestep)
        t0 = self.adaln_single.linear(t)
        y = self.caption_projection(y)

        x = auto_grad_checkpoint(self.transformer_blocks[0], x, None, y, None, t0, **kwargs)

        if c is not None:
            for index in range(1, self.copy_blocks_num + 1):
                before_proj, copied_block, after_proj = self.controlnet[index - 1]
                
                if index == 1:
                    c = before_proj(c) 
                
                c = auto_grad_checkpoint(copied_block, x + c, None, y, None, t0, **kwargs)
                c_skip = after_proj(c)
                x = auto_grad_checkpoint(self.transformer_blocks[index], x + c_skip, None, y, None, t0, **kwargs)
        
            for index in range(self.copy_blocks_num + 1, len(self.transformer_blocks)):
                x = auto_grad_checkpoint(self.transformer_blocks[index], x, None, y, None, t0, **kwargs)
        else:
            for index in range(1, len(self.transformer_blocks)):
                x = auto_grad_checkpoint(self.transformer_blocks[index], x, None, y, None, t0, **kwargs)

        x = self.norm_out(x)
        shift, scale = self.scale_shift_table.chunk(2, dim=1)
        x = x * (1 + scale.to(x.device)) + shift.to(x.device)
        x = self.proj_out(x)
        x = x.squeeze(1)
        x = x.reshape(shape=(-1, self.out_channels, self.height, self.width))
        return x
