_base_ = ['../PixelArt_xl2_internal.py']

# model setting
image_size = 256
window_block_indexes=[]
window_size=0
use_rel_pos=False
model = 'PixelArt_XL_2'
fp32_attention = True
load_from = "output/t2idit-xl2-img256_window16_pretrained_singlebr_part0-30_vae_lr2e5/checkpoints/epoch_80.pth"
vae_pretrained = "output/pretrained_models/sd-vae-ft-ema"
# training setting
eval_sampling_steps = 200

num_workers=10
train_batch_size = 176 # 32  # max 96 for DiT-L/4 when grad_checkpoint
num_epochs = 200 # 3
gradient_accumulation_steps = 1
grad_checkpointing = True
gradient_clip = 0.01
optimizer = dict(type='AdamW', lr=2e-5, weight_decay=3e-2, eps=1e-10)
lr_schedule_args = dict(num_warmup_steps=1000)

log_interval = 20
save_model_epochs=5
# work_dir = 'output/dit-xl2-img256_window16_pretrained'
work_dir = 'output/debug'
