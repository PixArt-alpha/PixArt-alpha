import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from pixart_controlnet_transformer import PixArtControlNetAdapterModel
from pipeline_pixart_alpha_controlnet import PixArtAlphaControlnetPipeline, get_closest_hw
import PIL.Image as Image

from pathlib import Path
import sys

current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent.parent))

from diffusion.model.hed import HEDdetector

output_dir = "output/controlnet"
Path(output_dir).mkdir(parents=True, exist_ok=True)

# input_image_path = "asset/images/controlnet/0_2.png"
# input_image_path = "asset/images/controlnet/battleship.jpg"
# input_image_path = "asset/images/controlnet/aidog.jpg"
input_image_path = "asset/images/controlnet/car.jpg"
given_image = Image.open(input_image_path)

# prompt = "A blue car, morning, city in background."
# prompt = "galaxy spaceship"
# prompt = "cute robot dog"
prompt = "modern car, city in background, morning, sunrise"

controlnet_strength = 1.0
weight_dtype = torch.float16
image_size = 1024

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(0)
# torch.manual_seed(1971901586)

# load controlnet
controlnet = PixArtControlNetAdapterModel.from_pretrained(
    "/home/raul/codelab/PixArt-alpha/ctrlnet/converted/controlnet",
    torch_dtype=weight_dtype,
    use_safetensors=True,
).to(device)

# print(controlnet.controlnet_blocks[0].after_proj.weight)
# print(controlnet.controlnet_blocks[0].after_proj.weight.dtype)
# print(controlnet.controlnet_blocks[0].after_proj.bias)
# print(controlnet.controlnet_blocks[0].after_proj.bias.dtype)
# exit()

pipe = PixArtAlphaControlnetPipeline.from_pretrained(
    "PixArt-alpha/PixArt-XL-2-1024-MS",
    controlnet=controlnet,
    torch_dtype=weight_dtype,
    use_safetensors=True,
).to(device)

# preprocess image, generate HED edge
hed = HEDdetector(False).to(device)

width, height = get_closest_hw(given_image.size[0], given_image.size[1], image_size)

condition_transform = T.Compose([
    T.Lambda(lambda img: img.convert('RGB')),
    T.Resize(int(min(height, width))),
    T.CenterCrop([int(height), int(width)]),
    T.ToTensor()
])

given_image = condition_transform(given_image).unsqueeze(0).to(device)
hed_edge = hed(given_image) * controlnet_strength
hed_edge = TF.normalize(hed_edge, [.5], [.5])
hed_edge = hed_edge.repeat(1, 3, 1, 1).to(weight_dtype)

# run pipeline
with torch.no_grad():
    out = pipe(
        prompt=prompt,
        image=hed_edge,
        num_inference_steps=14,
        guidance_scale=4.5,
        height=image_size,
        width=image_size,
    )

    out.images[0].save(f"{output_dir}/output.jpg")
    