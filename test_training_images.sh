CUDA_VISIBLE_DEVICES=5 python scripts/inference_ctrl_hed_train.py \
./configs/pixart_app_config/PixArt_xl2_img1024_controlHed_drop.py \
--model_path /home/xieenze/efs_cv/yue/controlnet_all_0.5data_drop0.3/checkpoints/epoch_2_step_5000.pth --port 12233 \
--test_mode train \
--exp_id controlnet_all_0.5data_drop0.3 \
--step 5000

CUDA_VISIBLE_DEVICES=6 python scripts/inference_ctrl_hed_train.py \
./configs/pixart_app_config/PixArt_xl2_img1024_controlHed_drop.py \
--model_path /home/xieenze/efs_cv/yue/controlnet_all_0.5data_drop0.3/checkpoints/epoch_3_step_10000.pth --port 12233 \
--test_mode train \
--exp_id controlnet_all_0.5data_drop0.3 \
--step 10000


CUDA_VISIBLE_DEVICES=7 python scripts/inference_ctrl_hed_train.py \
./configs/pixart_app_config/PixArt_xl2_img1024_controlHed_drop.py \
--model_path /home/xieenze/efs_cv/yue/controlnet_all_0.5data_drop0.3/checkpoints/epoch_5_step_15000.pth --port 12233 \
--test_mode train \
--exp_id controlnet_all_0.5data_drop0.3 \
--step 15000