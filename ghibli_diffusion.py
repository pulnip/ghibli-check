import PIL.Image
from diffusers import StableDiffusionPipeline
import torch

if __name__ == "__main__":
    import sys

    device = "mps" if torch.backends.mps.is_available() else \
            "cuda" if torch.cuda.is_available() else "cpu"

    prompt = "ghibli style magical castle in the sky, fantasy landscape, vibrant colors, whimsical characters"
    prompt = sys.argv[1] if len(sys.argv) > 1 else prompt

    model_id = "nitrosocke/Ghibli-Diffusion"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to(device)

    images: list[PIL.Image.Image] = pipe(prompt, num_images_per_prompt=1).images

    for i, image in enumerate(images):
        image.save(f"{prompt} {i}.png")
