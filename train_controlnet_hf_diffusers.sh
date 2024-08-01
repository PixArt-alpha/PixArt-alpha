#!/bin/bash

# run
# accelerate config

# check with
# accelerate env

MODEL_DIR="PixArt-alpha/PixArt-XL-2-1024-MS"
OUTPUT_DIR="./output/pixart-controlnet-open-pose"
TRAINING_DATA_DIR="/workspace/open_pose_controlnet"

accelerate launch ./controlnet/train_pixart_controlnet_hf.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --train_data_dir=$TRAINING_DATA_DIR \
 --resolution=1024 \
 --num_train_epochs=3 \
 --learning_rate=1e-5 \
 --train_batch_size=2 \
 --gradient_accumulation_steps=4 \
 --report_to="wandb" \
 --seed=42 \
 --dataloader_num_workers=8
#  --lr_scheduler="cosine" --lr_warmup_steps=0 \
#  --validation_image "./conditioning_image_1.png" "./conditioning_image_2.png" \
#  --validation_prompt "red circle with blue background" "cyan circle with brown floral background" \
