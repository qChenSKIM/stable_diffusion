import io
import base64
import os

import numpy as np
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline
from PIL import Image
from PIL import ImageOps
import gradio as gr
import base64
import skimage
import skimage.measure
from decouple import config

auth_token = config("HG_TOKEN")

# auth_token = os.getenv("auth_token")
model_id = "CompVis/stable-diffusion-v1-4"

try:
    cuda_available = torch.cuda.is_available()
except:
    cuda_available = False
finally:
    if cuda_available:
        device = "cuda"
    else:
        device = "cpu"

if device != "cuda":
    import contextlib
    autocast = contextlib.nullcontext
    

# pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=auth_token, revision="fp16", torch_dtype=torch.float16)
# pipe = pipe.to(device)

if device=="cuda":
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        revision="fp16",
        torch_dtype=torch.float16,
        use_auth_token=auth_token,
    ).to(device)
else:
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        use_auth_token=auth_token,
    ).to(device)
            

def infer(prompt, steps, scale, seed):        
    generator = torch.Generator(device=device).manual_seed(seed)
    images_list = pipe(
        [prompt] * 1,
        num_inference_steps=steps,
        guidance_scale=scale,
        generator=generator,
    )
    images = []
    safe_image = Image.open(r"unsafe.png")
    for i, image in enumerate(images_list["sample"]):
        if(images_list["nsfw_content_detected"][i]):
            images.append(safe_image)
        else:
            images.append(image)
    return images
    


block = gr.Blocks()

with block:
    with gr.Group():
        with gr.Box():
            with gr.Row().style(mobile_collapse=False, equal_height=True):
                text = gr.Textbox(
                    label="Enter your prompt",
                    show_label=False,
                    max_lines=1,
                    placeholder="Enter your prompt",
                ).style(
                    border=(True, False, True, True),
                    rounded=(True, False, False, True),
                    container=False,
                )
                btn = gr.Button("Generate image").style(
                    margin=False,
                    rounded=(False, True, True, False),
                )
        gallery = gr.Gallery(
            label="Generated images", show_label=False, elem_id="gallery"
        ).style(grid=[2], height="auto")

        advanced_button = gr.Button("Advanced options", elem_id="advanced-btn")

        with gr.Row(elem_id="advanced-options"):
            # samples = gr.Slider(label="Images", minimum=1, maximum=1, value=1, step=1)
            steps = gr.Slider(label="Steps", minimum=1, maximum=50, value=10, step=1)
            scale = gr.Slider(
                label="Closness to text prompt (Guidance Scale)", minimum=0, maximum=50, value=7.5, step=0.1
            )
            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=2147483647,
                step=1,
                randomize=True,
            )
        # text.submit(infer, inputs=[text, samples, steps, scale, seed], outputs=gallery)
        text.submit(infer, inputs=[text, steps, scale, seed], outputs=gallery)
        btn.click(infer, inputs=[text, steps, scale, seed], outputs=gallery)
        advanced_button.click(
            None,
            [],
            text,
        )
        
block.launch(max_threads=150,server_name="0.0.0.0", server_port=5000,share=True)
