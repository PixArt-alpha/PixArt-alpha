#!/bin/bash

# in mixed-precision mode, the the training script cannot validate the model, so we need to run the validation script separately

CHECKPOINTS_FOLDER="output/pixart-controlnet-hf-diffusers-test"
OUTPUT_DIR="output/controlnet/validation_hf_diffusers"
CONTROL_IMAGES_FOLDER="output/controlnet/control_images"

python ./controlnet/validate_folder_with_checkpoints.py \
 --checkpoints_folder "$CHECKPOINTS_FOLDER" \
 --output_folder "$OUTPUT_DIR" \
 --control_images_folder "CONTROL_IMAGES_FOLDER"