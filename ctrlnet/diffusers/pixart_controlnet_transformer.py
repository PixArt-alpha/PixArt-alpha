import torch
from torch import nn

from .pixart_transformer_2d import PixArtTransformer2DModel
from diffusion.model.utils import auto_grad_checkpoint


class PixArtControlNetTransformerModel(PixArtTransformer2DModel):
    def __init__(self, *args, blocks_num=13, **kwargs):
        super().__init__(*args, **kwargs)

        self.blocks_num = blocks_num
        self.transformer_blocks = nn.ModuleList()
        
        # Copy first copy_blocks_num transformer blocks for ControlNet
        for i in range(blocks_num):
            self.controlnet.append(nn.ModuleList([
                nn.Linear(self.inner_dim, self.inner_dim),  # before_proj
                deepcopy(self.transformer_blocks[i]),       # copied_block 
                nn.Linear(self.inner_dim, self.inner_dim)   # after_proj
            ]))
            
            # Zero init before/after proj
            nn.init.zeros_(self.controlnet[i][0].weight)
            nn.init.zeros_(self.controlnet[i][0].bias)
            nn.init.zeros_(self.controlnet[i][2].weight) 
            nn.init.zeros_(self.controlnet[i][2].bias)

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
