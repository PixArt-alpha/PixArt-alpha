import os
import sys
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image

script_dir = os.path.dirname(os.path.abspath(__file__))
subfolder_path = os.path.join(script_dir, 'pipeline')
sys.path.insert(0, subfolder_path)

from pipeline.pixart_controlnet_transformer import PixArtControlNetAdapterModel
from pipeline.pipeline_pixart_alpha_controlnet import PixArtAlphaControlnetPipeline, get_closest_hw

# MODEL_ID="PixArt-alpha/PixArt-XL-2-1024-MS"
MODEL_ID="PixArt-alpha/PixArt-XL-2-512x512"

def generate_images_from_checkpoints(checkpoints_folder, output_folder, prompts, control_images, image_size=1024, weight_dtype=torch.float16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    validation_images = []
    for control_image in control_images:
        validation_image = Image.open(control_image).convert("RGB")
        validation_image = validation_image.resize((image_size, image_size))
        validation_images.append(validation_image)

    if len(validation_images) == 0:
        print("Validation images are empty.")
        return

    print(f"Validation images: {len(validation_images)}")

    for folder in os.listdir(checkpoints_folder):
        checkpoint_folder = os.path.join(checkpoints_folder, folder, "controlnet")

        if os.path.isdir(checkpoint_folder):
            print(f"Found checkpoint from {checkpoint_folder}")

            checkpoint_number = folder.split('-')[-1]

            controlnet = PixArtControlNetAdapterModel.from_pretrained(
                checkpoint_folder,
                torch_dtype=weight_dtype,
                use_safetensors=True,
            ).to(device)

            pipe = PixArtAlphaControlnetPipeline.from_pretrained(
                MODEL_ID,
                controlnet=controlnet,
                torch_dtype=weight_dtype,
                use_safetensors=True,
            ).to(device)

            for i, prompt in enumerate(prompts):
                with torch.no_grad():
                    out = pipe(
                        prompt=prompt,
                        image=validation_images[i],
                        num_inference_steps=14,
                        guidance_scale=4.5,
                        height=image_size,
                        width=image_size,
                    )
                    
                    output_image_path = os.path.join(output_folder, f"{checkpoint_number}_img_{i+1}.jpg")
                    out.images[0].save(output_image_path)

                    print(f"\tSaved image to {output_image_path}")

            print(f"  Finished processing checkpoint {checkpoint_folder}!")

if __name__ == "__main__":
    checkpoints_folder = "path/to/checkpoints"
    output_folder = "path/to/output"
    path_to_control_images = "path/to/control_images"

    prompts = ["red circle with blue background", "cyan circle with brown floral background"]
    control_images = [
        os.path.join(path_to_control_images, "conditioning_image_1.png"),
        os.path.join(path_to_control_images, "conditioning_image_2.png")
    ]

    assert len(prompts) == len(control_images)

    generate_images_from_checkpoints(checkpoints_folder, output_folder, prompts, control_images, image_size=512, weight_dtype=torch.float16)
