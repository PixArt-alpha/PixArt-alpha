#!/bin/bash

# run
# accelerate config

# check with
# accelerate env

export MODEL_DIR="PixArt-alpha/PixArt-XL-2-512x512"
export OUTPUT_DIR="pixart-controlnet-hf-diffusers-test"

accelerate launch ./controlnet/train_pixart_controlnet_hf.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name=fusing/fill50k \
 --mixed_precision="fp16" \
 --resolution=512 \
 --learning_rate=1e-5 \
 --validation_image "./conditioning_image_1.png" "./conditioning_image_2.png" \
 --validation_prompt "red circle with blue background" "cyan circle with brown floral background" \
 --train_batch_size=1 \
 --gradient_accumulation_steps=4 \
 --report_to="wandb" \
 --seed=42 \
 --dataloader_num_workers=8
#  --lr_scheduler="cosine" --lr_warmup_steps=0 \
