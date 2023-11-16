<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

[//]: # (&#40;reference from [hugging Face]&#40;https://github.com/huggingface/diffusers/blob/docs/8bit-inference-pixart/docs/source/en/api/pipelines/pixart.md&#41;&#41;)

## Running the `PixArtAlphaPipeline` in under 8GB GPU VRAM

It is possible to run the [`PixArtAlphaPipeline`] under 8GB GPU VRAM by loading the text encoder in 8-bit numerical precision. Let's walk through a full-fledged example. 

First, install the `bitsandbytes` library:

```bash
pip install -U bitsandbytes
```

Then load the text encoder in 8-bit:

```python
from transformers import T5EncoderModel
from diffusers import PixArtAlphaPipeline

text_encoder = T5EncoderModel.from_pretrained(
    "PixArt-alpha/PixArt-XL-2-1024-MS",
    subfolder="text_encoder",
    load_in_8bit=True,
    device_map="auto",

)
pipe = PixArtAlphaPipeline.from_pretrained(
    "PixArt-alpha/PixArt-XL-2-1024-MS",
    text_encoder=text_encoder,
    transformer=None,
    device_map="auto"
)
```

Now, use the `pipe` to encode a prompt:

```python
with torch.no_grad():
    prompt = "cute cat"
    prompt_embeds, prompt_attention_mask, negative_embeds, negative_prompt_attention_mask = pipe.encode_prompt(prompt)

del text_encoder
del pipe
flush()
```

`flush()` is just a utility function to clear the GPU VRAM and is implemented like so:

```python
import gc 

def flush():
    gc.collect()
    torch.cuda.empty_cache()
```

Then compute the latents providing the prompt embeddings as inputs:

```python
pipe = PixArtAlphaPipeline.from_pretrained(
    "PixArt-alpha/PixArt-XL-2-1024-MS",
    text_encoder=None,
    torch_dtype=torch.float16,
).to("cuda")

latents = pipe(
    negative_prompt=None, 
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_embeds,
    prompt_attention_mask=prompt_attention_mask,
    negative_prompt_attention_mask=negative_prompt_attention_mask,
    num_images_per_prompt=1,
    output_type="latent",
).images

del pipe.transformer
flush()
```

Notice that while initializing `pipe`, you're setting `text_encoder` to `None` so that it's not loaded. 

Once the latents are computed, pass it off the VAE to decode into a real image:

```python
with torch.no_grad():
    image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
image = pipe.image_processor.postprocess(image, output_type="pil")
image.save("cat.png")
```

All of this, put together, should allow you to run [`PixArtAlphaPipeline`] under 8GB GPU VRAM.

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/pixart/8bits_cat.png)

Find the script [here](https://gist.github.com/sayakpaul/3ae0f847001d342af27018a96f467e4e) that can be run end-to-end to report the memory being used.

<Tip warning={true}>

Text embeddings computed in 8-bit can have an impact on the quality of the generated images because of the information loss in the representation space induced by the reduced precision. It's recommended to compare the outputs with and without 8-bit.

</Tip>