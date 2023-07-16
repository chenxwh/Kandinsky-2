# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
from typing import List
import numpy as np
import torch
from transformers import pipeline
from diffusers.utils import load_image
from diffusers import (
    KandinskyV22PriorPipeline,
    KandinskyV22ControlnetPipeline,
    KandinskyV22PriorEmb2EmbPipeline,
    KandinskyV22ControlnetImg2ImgPipeline,
)

from cog import BasePredictor, Input, Path


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.depth_estimator = pipeline("depth-estimation")
        self.pipe_prior = KandinskyV22PriorPipeline.from_pretrained(
            "kandinsky-community/kandinsky-2-2-prior",
            torch_dtype=torch.float16,
            cache_dir="weights_cache_t2i",
        ).to("cuda")
        self.pipe = KandinskyV22ControlnetPipeline.from_pretrained(
            "kandinsky-community/kandinsky-2-2-controlnet-depth",
            torch_dtype=torch.float16,
            cache_dir="weights_cache_t2i",
        ).to("cuda")
        self.pipe_prior_img2img = KandinskyV22PriorEmb2EmbPipeline.from_pretrained(
            "kandinsky-community/kandinsky-2-2-prior",
            torch_dtype=torch.float16,
            cache_dir="weights_cache_i2i",
        ).to("cuda")
        self.pipe_img2img = KandinskyV22ControlnetImg2ImgPipeline.from_pretrained(
            "kandinsky-community/kandinsky-2-2-controlnet-depth",
            torch_dtype=torch.float16,
            cache_dir="weights_cache_i2i",
        ).to("cuda")

    def predict(
        self,
        image: Path = Input(description="Input image", default=None),
        prompt: str = Input(
            description="Input prompt",
            default="A robot, 4k photo",
        ),
        negative_prompt: str = Input(
            description="Specify things to not see in the output",
            default="lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature",
        ),
        task: str = Input(
            default="img2img",
            choices=["text2img", "img2img"],
            description="Choose a task",
        ),
        width: int = Input(
            description="Width of output image. Lower the setting if hits memory limits.",
            choices=[
                384,
                512,
                576,
                640,
                704,
                768,
                960,
                1024,
                1152,
                1280,
                1536,
                1792,
                2048,
            ],
            default=768,
        ),
        height: int = Input(
            description="Height of output image. Lower the setting if hits memory limits.",
            choices=[
                384,
                512,
                576,
                640,
                704,
                768,
                960,
                1024,
                1152,
                1280,
                1536,
                1792,
                2048,
            ],
            default=768,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=75
        ),
        num_outputs: int = Input(
            description="Number of images to output.",
            ge=1,
            le=4,
            default=1,
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")
        generator = torch.Generator(device="cuda").manual_seed(seed)

        img = load_image(str(image)).resize((width, height))

        hint = make_hint(img, self.depth_estimator).unsqueeze(0).half().to("cuda")

        if task == "img2img":
            img_emb = self.pipe_prior_img2img(
                prompt=prompt, image=img, strength=0.85, generator=generator
            )
            negative_emb = self.pipe_prior_img2img(
                prompt=negative_prompt, image=img, strength=1, generator=generator
            )
            images = self.pipe_img2img(
                image=img,
                strength=0.5,
                image_embeds=img_emb.image_embeds,
                negative_image_embeds=negative_emb.image_embeds,
                hint=hint,
                num_inference_steps=num_inference_steps,
                generator=generator,
                height=height,
                width=width,
            ).images

        else:
            image_emb, zero_image_emb = self.pipe_prior(
                prompt=prompt, negative_prompt=negative_prompt, generator=generator
            ).to_tuple()

            images = self.pipe(
                image_embeds=image_emb,
                negative_image_embeds=zero_image_emb,
                hint=hint,
                num_inference_steps=num_inference_steps,
                generator=generator,
                height=height,
                width=width,
            ).images

        output_paths = []
        for i, sample in enumerate(images):
            output_path = f"/tmp/out-{i}.png"
            sample.save(output_path)
            output_paths.append(Path(output_path))

        return output_paths


# let's take an image and extract its depth map.
def make_hint(image, depth_estimator):
    image = depth_estimator(image)["depth"]
    image = np.array(image)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    detected_map = torch.from_numpy(image).float() / 255.0
    hint = detected_map.permute(2, 0, 1)
    return hint
