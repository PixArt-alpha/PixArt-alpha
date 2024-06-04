
```python
import torch
from scripts.pipeline_pixart_inpaint import PixArtAlphaInpaintPipeline
from PIL import Image

pipe = PixArtAlphaInpaintPipeline.from_pretrained("PixArt-alpha/PixArt-XL-2-1024-MS", torch_dtype=torch.float16)

prompt = ""
image = Image.open('')
mask_image = Image.open('')
out = pipe(prompt, image=image, mask_image=mask_image, strength=1.0).images[0]
out.save('./cactus_removed.png')
```