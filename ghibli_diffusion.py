from numpy.random import choice
from numpy import int64
import os
from typing import Literal
from PIL.Image import Image
from diffusers import StableDiffusionPipeline as SDP
import torch

DEVICE = "mps" if torch.backends.mps.is_available() else \
        "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "nitrosocke/Ghibli-Diffusion"
pipe = SDP.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(DEVICE)

def gen_and_save_image(
    prompt: str,
    num_images: str,
    out_dir: str
):
    ghibli_prompt = f"ghibli style, {prompt}"

    images: list[Image] = pipe(ghibli_prompt, num_images_per_prompt=num_images).images

    counter = 0
    while os.path.exists(f"{out_dir}/{prompt} {counter}.png"):
        counter += 1
    for image in images:
        image.save(f"{out_dir}/{prompt} {counter}.png")
        counter += 1

def get_prompts(
    prompt_filename: str,
    num_prompt: int,
    order: Literal["sequential", "random"]
):
    prompts = []

    with open(prompt_filename) as f:
        all_prompts = f.readlines()

        if order == "sequential":
            prompts = all_prompts[0:num_prompt]
        else:
            indexes = choice(range(0, len(all_prompts)), replace=False)
            prompts = [all_prompts[indexes]] if type(indexes) == int64 \
                    else [all_prompts[i] for i in indexes]

    return prompts

def gen_image_from_prompts(
    num_prompt: int,
    images_per_prompt: int,
    order: Literal["sequential", "random"] = "random",
    prompt_filename: str = "on_theme.txt",
    image_outdir: str = "on_theme"
):
    prompts = get_prompts(prompt_filename, num_prompt, order)
    for prompt in prompts:
        print(f"Prompt: {prompt}")
        gen_and_save_image(prompt, images_per_prompt, image_outdir)

if __name__ == "__main__":
    import sys

    num_prompt = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    images_per_prompt = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    order = sys.argv[3] if len(sys.argv) > 3 else "sequential"
    assert order == "sequential" or order == "random"

    gen_image_from_prompts(num_prompt, images_per_prompt, order,
        "on_theme.txt", "on_theme")
    # gen_image_from_prompts(num_prompt, images_per_prompt, order,
    #     "off_theme.txt", "off_theme")
