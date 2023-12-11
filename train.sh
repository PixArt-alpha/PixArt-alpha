# CUDA_VISIBLE_DEVICES=5,6,7 python -m torch.distributed.launch --nproc_per_node=3 \
#     --master_port=26662 train_scripts/train_controlnet.py \
#     ./configs/pixart_app_config/PixArt_xl2_img512_controlHed.py \
#     --work-dir output_debug \
#     --controlnet_type half
    # --resume_from '/home/xieenze/efs_cv/yue/controlnet_all_0.5data_drop0.3_lr2e-5/checkpoints/epoch_7_step_22000.pth' \

# CUDA_VISIBLE_DEVICES=5,6,7 python -m torch.distributed.launch --nproc_per_node=3 \
#     --master_port=26662 train_scripts/train_controlnet.py \
#     ./configs/pixart_app_config/PixArt_xl2_img1024_controlHed_drop.py \
#     --work-dir output_debug \
#     --controlnet_type all \
#     --resume_from '/home/xieenze/efs_cv/yue/controlnet_all_0.5data_drop0.3_lr2e-5/checkpoints/epoch_7_step_22000.pth' \
#     --resume_optimizer \
#     --resume_lr_scheduler