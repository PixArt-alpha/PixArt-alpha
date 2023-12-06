_base_ = ['../PixArt_xl2_internal.py']
data_root = 'data'

# data = dict(type='MJHed', root='MJData', image_list_json=['mj_4_new_debug.json'], transform='default_train', load_vae_feat=True)
data = dict(type='MJHed', root='MJData', image_list_json=['mj_1_new.json'], transform='default_train', load_vae_feat=True)
# model setting
image_size = 1024
window_block_indexes = []
window_size=0
use_rel_pos=False
model = 'PixArtMS_XL_2'
fp32_attention = True
# load_from = 'output_cv/controlnet_all/checkpoints/epoch_1_step_2000.pth'
load_from = 'pretrained_pixart/1024/epoch_1_step_22000.pth'
# load_from = './output_all/checkpoints/epoch_1_step_1000.pth'
# load_from = None
vae_pretrained = "output/pretrained_models/sd-vae-ft-ema"
lewei_scale = 2.0

# training setting
use_fsdp=False   # if use FSDP mode
num_workers=10
train_batch_size = 2 # 32  # max 96 for DiT-L/4 when grad_checkpoint
num_epochs = 100 # 3
gradient_accumulation_steps = 1
grad_checkpointing = True
gradient_clip = 0.01
optimizer = dict(type='AdamW', lr=5e-6, weight_decay=3e-2, eps=1e-10)
lr_schedule_args = dict(num_warmup_steps=0)

eval_sampling_steps = 200
log_interval = 20
save_model_epochs=5
save_model_steps=1000
work_dir = 'output_debug/debug'
class_dropout_prob = 0.3
train_ratio = 0.5
