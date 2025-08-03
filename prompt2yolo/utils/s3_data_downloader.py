import logging
from typing import List, Optional

from prompt2yolo.utils.logger import setup_logger
from prompt2yolo.utils.s3_handler import S3Handler


class S3DataDownloader:
    def __init__(
        self,
        s3_image_folder: str,
        s3_label_folder: str,
        image_folder: str,
        label_folder: str,
        logger: Optional[logging.Logger] = None,
    ):
        self.s3_image_folder = s3_image_folder
        self.s3_label_folder = s3_label_folder
        self.image_folder = image_folder
        self.label_folder = label_folder

        # Create S3Handler instances for images and labels
        self.image_downloader = S3Handler(
            s3_folder=s3_image_folder, local_dir=image_folder, logger=logger
        )
        self.label_downloader = S3Handler(
            s3_folder=s3_label_folder, local_dir=label_folder, logger=logger
        )
        self.logger = logger or setup_logger(__name__)

    def download_images_and_labels(
        self,
        image_extensions: List[str] = ["jpg", "png"],
        label_extensions: List[str] = ["txt"],
    ):
        """Download images and labels from S3 to the desired local paths."""
        self.logger.info(f"[*] Downloading image and labels")

        self.image_downloader.download_files_by_extension(
            file_extensions=image_extensions
        )
        self.label_downloader.download_files_by_extension(
            file_extensions=label_extensions
        )


class S3ModelDownloader:
    def __init__(
        self,
        s3_model_folder: str,
        model_folder: str,
        logger: Optional[logging.Logger] = None,
    ):
        self.s3_model_folder = s3_model_folder
        self.model_folder = model_folder

        # Create S3Handler instance for the model folder
        self.model_downloader = S3Handler(
            s3_folder=s3_model_folder, local_dir=model_folder, logger=logger
        )
        self.logger = logger or setup_logger(__name__)

    def download_model_files(
        self,
        model_extensions: List[str] = ["pt", "onnx", "h5"],
    ):
        """Download model files from S3 to the desired local path."""
        self.logger.info("[*] Downloading models.")
        self.model_downloader.download_files_by_extension(
            file_extensions=model_extensions
        )

    def download_entire_folder(self):
        """Download all files from the specified S3 folder to the local path."""
        self.logger.info(f"[*] Downloading entire folder: {self.s3_model_folder}")
        self.model_downloader.download_all_files()
