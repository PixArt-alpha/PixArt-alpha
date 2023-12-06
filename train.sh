CUDA_VISIBLE_DEVICES=0,2,3,4 python -m torch.distributed.launch --nproc_per_node=4 --master_port=26662 train_scripts/train_controlnet.py \
    ./configs/pixart_app_config/PixArt_xl2_img1024_controlHed.py \
    --work-dir output_all