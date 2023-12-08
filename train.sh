CUDA_VISIBLE_DEVICES=2,5 python -m torch.distributed.launch --nproc_per_node=2 --master_port=26662 train_scripts/train_controlnet.py \
    ./configs/pixart_app_config/PixArt_xl2_img1024_controlHed_drop.py \
    --resume_from '/home/xieenze/efs_cv/yue/controlnet_all/checkpoints/epoch_1_step_2000.pth' \
    --work-dir output_all