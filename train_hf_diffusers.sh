#!/bin/bash

# run
# accelerate config

# check with
# accelerate env

# remove the validation_epochs or set it to a lower number if you want to run the validation prompt
# validation will be ran at the end
accelerate launch --num_processes=1 ./train_scripts/train_pixart_hf.py --mixed_precision="bf16" \
  --pretrained_model_name_or_path=PixArt-alpha/PixArt-XL-2-1024-MS \
  --train_data_dir="../PixArt-alpha-finetuning/data/lego-city-adventures-captions/" --caption_column="llava_caption_with_orig_caption" \
  --resolution=1024 \
  --train_batch_size=2 --gradient_accumulation_steps=1 \
  --num_train_epochs=100 --checkpointing_steps=200 \
  --checkpoints_total_limit=30 \
  --learning_rate=3e-04 --lr_scheduler="cosine" --lr_warmup_steps=0 \
  --seed=42 \
  --output_dir="lego-city-adventures-model" \
  --report_to="wandb" \
  --gradient_checkpointing \
  --validation_epochs=5 \
  --validation_prompt="Image in lego city adventures style, cute dragon creature" \
  --adam_weight_decay=0.03 --adam_epsilon=1e-10 \
  --dataloader_num_workers=8
  # --max_train_samples=300 \
  # --snr_gamma=1.0
