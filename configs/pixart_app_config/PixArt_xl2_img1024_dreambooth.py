_base_ = ['../PixArt_xl2_internal.py']
data_root = 'data/dreambooth/dataset'

data = dict(type='DreamBooth', root='dog6', prompt=['a photo of sks dog'], transform='default_train', load_vae_feat=True)
image_size = 1024

# model setting
model = 'PixArtMS_XL_2'     # model for multi-scale training
fp32_attention = True
load_from = 'Path/to/PixArt-XL-2-1024-MS.pth'
vae_pretrained = "output/pretrained_models/sd-vae-ft-ema"
window_block_indexes = []
window_size=0
use_rel_pos=False
aspect_ratio_type = 'ASPECT_RATIO_1024'         # base aspect ratio [ASPECT_RATIO_512 or ASPECT_RATIO_256]
multi_scale = True     # if use multiscale dataset model training
lewei_scale = 2.0

# training setting
num_workers=1
train_batch_size = 1
num_epochs = 200
gradient_accumulation_steps = 1
grad_checkpointing = True
gradient_clip = 0.01
optimizer = dict(type='AdamW', lr=5e-6, weight_decay=3e-2, eps=1e-10)
lr_schedule_args = dict(num_warmup_steps=0)
auto_lr = None

log_interval = 1
save_model_epochs=10000
save_model_steps=100
work_dir = 'output/debug'
