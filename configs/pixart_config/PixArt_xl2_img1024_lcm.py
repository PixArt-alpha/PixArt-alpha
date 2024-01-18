_base_ = ['../PixArt_xl2_internal.py']
data_root = 'data'
image_list_json = ['data_info.json',]

data = dict(type='InternalDataMS', root='InternData', image_list_json=image_list_json, transform='default_train', load_vae_feat=True)
image_size = 1024

# model setting
model = 'PixArtMS_XL_2'     # model for multi-scale training
fp32_attention = False  # Set to True if you got NaN loss
load_from = None
vae_pretrained = "output/pretrained_models/sd-vae-ft-ema"
window_block_indexes = []
window_size=0
use_rel_pos=False
aspect_ratio_type = 'ASPECT_RATIO_1024'         # base aspect ratio [ASPECT_RATIO_512 or ASPECT_RATIO_256]
multi_scale = True     # if use multiscale dataset model training
lewei_scale = 2.0

# training setting
num_workers=4
train_batch_size = 16   # max 12 for PixArt-xL/2 when grad_checkpoint   16 for LCM-LoRA
num_epochs = 10 # 3
gradient_accumulation_steps = 1
grad_checkpointing = True
gradient_clip = 0.01
optimizer = dict(type='AdamW', lr=2e-5, weight_decay=0.0, eps=1e-10)
# optimizer = dict(type='CAMEWrapper', lr=1e-7, weight_decay=0.0, betas=(0.9, 0.999, 0.9999), eps=(1e-30, 1e-16))
lr_schedule_args = dict(num_warmup_steps=100)
save_model_epochs=1
save_model_steps=200
valid_num=0      # take as valid aspect-ratio when sample number >= valid_num

log_interval = 10
eval_sampling_steps = 200
work_dir = 'output/debug'

# LCM
loss_type = 'huber'
huber_c = 0.001
num_ddim_timesteps=50
w_max = 15.0
w_min = 3.0
ema_decay = 0.95
cfg_scale = 4.5
class_dropout_prob = 0.
lora_rank = 32