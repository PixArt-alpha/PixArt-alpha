from mmcv import Registry

from diffusion.model.utils import set_grad_checkpoint

MODELS = Registry('models')


def build_model(cfg, use_grad_checkpoint=False, use_fp32_attention=False, gc_step=1, **kwargs):
    if isinstance(cfg, str):
        cfg = dict(type=cfg)
    model = MODELS.build(cfg, default_args=kwargs)
    if use_grad_checkpoint:
        set_grad_checkpoint(model, use_fp32_attention=use_fp32_attention, gc_step=gc_step)
    return model
