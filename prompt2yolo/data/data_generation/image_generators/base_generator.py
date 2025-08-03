import logging
import os
from pathlib import Path
from typing import Optional

from botocore.exceptions import ClientError
from diffusers import DiffusionPipeline
from dotenv import load_dotenv

from prompt2yolo.utils.logger import setup_logger

load_dotenv()

from prompt2yolo.configs import ImageGeneratorConfig, Paths
from prompt2yolo.utils.s3_handler import S3Handler

LOCAL_IMAGE_FOLDER = Paths().image_folder


class DreamBoothGeneratorBase:
    def __init__(
        self,
        s3_handler: S3Handler,
        config: ImageGeneratorConfig,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.s3_handler = s3_handler
        self.config = config
        self.logger = logger or setup_logger(__name__)

        self.s3_base_folder = Paths().s3_image_folder
        self.lora_path = self.download_lora_checkpoint()

        # Load the model
        self.pipe = self.load_model()

    def download_lora_checkpoint(self) -> str:
        """Download all relevant LoRA checkpoint files from S3 and return the main checkpoint path."""
        # List all files in the specified S3 folder
        files = self.s3_handler.list_files_in_folder()

        if not files:
            error_msg = f"No files found in '{self.s3_handler.s3_folder}'."
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        self.s3_handler.logger.debug(f"Files returned from S3: {files}")

        downloaded_files = []

        # Ensure the local directory exists
        local_dir = Path(self.s3_handler.local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)

        # Download each file to the local directory
        for file in files:
            if not file.strip():
                self.logger.info("Skipping empty file entry.")
                continue

            filename = os.path.basename(file)
            if not filename:
                self.logger.info(f"Skipping entry with empty filename: {file}")
                continue

            # Skip hidden or special files
            if filename.startswith("."):
                self.logger.info(f"Skipping hidden or special file: {filename}")
                continue

            local_path = local_dir / filename

            try:
                # Use download_file to download directly to a local path
                self.s3_handler.s3_client.download_file(
                    self.s3_handler.bucket_name, file, str(local_path)
                )
                self.logger.info(f"Successfully downloaded {file} to {local_path}")
                downloaded_files.append(str(local_path))
            except ClientError as e:
                error_msg = f"Failed to download {file}: {e}"
                self.logger.error(error_msg)
                raise ValueError(error_msg) from e

        # Find the specific LoRA weights file (e.g., pytorch_lora_weights.safetensors)
        lora_file = next(
            (
                f
                for f in downloaded_files
                if "pytorch_lora_weights" in f and f.endswith(".safetensors")
            ),
            None,
        )

        if not lora_file:
            error_msg = (
                "No valid LoRA weights file (e.g., 'pytorch_lora_weights.safetensors') "
                "found among the downloaded files."
            )
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        # Return the path of the main LoRA weights file
        return lora_file

    def sanitize_filename(self, filename: str, max_length: int = 75) -> str:
        invalid_chars = ["<", ">", ":", '"', "/", "\\", "|", "?", "*", ",", " "]
        for char in invalid_chars:
            filename = filename.replace(char, "_")
        return filename[:max_length]

    def load_model(self) -> DiffusionPipeline:
        """Load the diffusion model with the optional LoRA checkpoint(s)."""
        raise NotImplementedError("This method must be implemented in a subclass.")

    def generate(
        self, prompt: str, weight: float, output_dir: str = LOCAL_IMAGE_FOLDER
    ) -> None:
        raise NotImplementedError("This method must be implemented in a subclass.")
