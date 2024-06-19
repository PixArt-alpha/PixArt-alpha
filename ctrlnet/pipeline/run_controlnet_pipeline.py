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

input_image_path = "asset/images/controlnet/0_2.png"
given_image = Image.open(input_image_path)

prompt = "A Electric 4 seats mini VAN,simple design stylel,led headlight,front 45 angle view,sunlight,clear sky."

controlnet_strength = 1.0
weight_dtype = torch.float16
image_size = 1024

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(0)

# load controlnet
controlnet = PixArtControlNetAdapterModel()
controlnet.from_pretrained("/home/raul/codelab/PixArt-alpha/ctrlnet/converted/controlnet")

pipe = PixArtAlphaControlnetPipeline.from_pretrained(
    "PixArt-alpha/PixArt-XL-2-1024-MS",
    controlnet=controlnet,
    torch_dtype=weight_dtype,
    use_safetensors=True,
).to(device)

# preprocess image, generate HED edge
hed = HEDdetector(False).to(device)

closest_hw = get_closest_hw(given_image.size[0], given_image.size[1], image_size)

condition_transform = T.Compose([
    T.Lambda(lambda img: img.convert('RGB')),
    T.Resize(int(min(closest_hw))),
    T.CenterCrop([int(closest_hw[0]), int(closest_hw[1])])
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
        num_inference_steps=30,
        guidance_scale=4.5,
        height=image_size,
        width=image_size,
    )

    out.save("output/controlnet/output.jpg")