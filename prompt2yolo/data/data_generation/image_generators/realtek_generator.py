from pathlib import Path

import torch
from diffusers import AutoencoderKL, DiffusionPipeline

from prompt2yolo.configs import Paths
from prompt2yolo.data.data_generation.image_generators.base_generator import (
    DreamBoothGeneratorBase,
)


class RealtekGenerator(DreamBoothGeneratorBase):
    def load_model(self) -> DiffusionPipeline:
        """Realtek-specific LoRA loading logic."""
        try:
            vae = AutoencoderKL.from_pretrained(
                self.config.vae_path, torch_dtype=torch.float16
            )

            pipe = DiffusionPipeline.from_pretrained(
                self.config.model_path,
                vae=vae,
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True,
            )
            if self.lora_path:
                pipe.load_lora_weights(self.lora_path)

            pipe = pipe.to("cuda")

        except Exception as e:
            self.s3_handler.logger.error(f"Failed to load Realtek model: {e}")
            raise RuntimeError(f"Failed to load Realtek model: {e}") from e

        return pipe

    def generate(
        self, prompt: str, weight: float, output_dir: str = Paths().image_folder
    ) -> None:
        """Generate images without grayscale conversion for Realtek."""
        num_images = max(1, int(weight * self.config.num_images))
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        seeds = [
            torch.randint(1000000000000, 9999999999999, (1,)).item()
            for _ in range(num_images)
        ]
        sanitized_prompt = self.sanitize_filename(prompt)

        for seed in seeds:
            generator = torch.Generator("cuda").manual_seed(seed)
            try:
                result = self.pipe(
                    prompt=prompt,
                    negative_prompt=self.config.negative_prompt,
                    generator=generator,
                    height=self.config.image_size[1],
                    width=self.config.image_size[0],
                    num_images_per_prompt=1,
                    num_inference_steps=self.config.steps,
                    guidance_scale=self.config.guidance_scale,
                )
                image = result.images[0]

                image_filename = f"{sanitized_prompt}_{seed}.jpg"
                image_path = output_path / image_filename
                image.save(image_path)
            except Exception as e:
                self.s3_handler.logger.error(f"Failed to generate image: {e}")

        torch.cuda.empty_cache()
