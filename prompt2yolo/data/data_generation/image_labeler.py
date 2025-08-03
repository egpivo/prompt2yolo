import logging
import os
import shutil
import tempfile  # Import tempfile for secure temporary directory handling
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()

from ultralytics import YOLO

from prompt2yolo.configs import Paths, YoloLabelerConfig
from prompt2yolo.utils.logger import setup_logger
from prompt2yolo.utils.s3_handler import S3Handler

LOCAL_IMAGE_FOLDER = Paths().image_folder
LOCAL_LABEL_FOLDER = Paths().label_folder


class YoloWorldLabeler:
    def __init__(
        self,
        s3_handler: S3Handler,
        custom_classes: List[str],
        config: YoloLabelerConfig,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.s3_handler = s3_handler
        self.custom_classes = custom_classes
        self.config = config
        self.model = self._initialize_model()
        self.logger = logger or setup_logger()

        # Create a secure, unique temporary directory
        self._run_path = tempfile.mkdtemp(prefix="yolo_run_")

    def _initialize_model(self) -> YOLO:
        """Private method to initialize and customize the YOLO model."""
        model = YOLO(self.config.yolo_model)
        model.set_classes(self.custom_classes)
        return model

    def _prepare_run_directory(self) -> None:
        """Create or clear the temporary run directory."""
        # Clear contents if the directory already exists
        if os.path.exists(self._run_path):
            shutil.rmtree(self._run_path)
        os.makedirs(self._run_path, exist_ok=True)

    def label(
        self,
        local_image_path: str = LOCAL_IMAGE_FOLDER,
        output_dir: str = LOCAL_LABEL_FOLDER,
    ) -> None:
        self._prepare_run_directory()

        # Run the YOLO model prediction
        _ = self.model.predict(
            local_image_path,
            save=True,
            save_txt=True,
            imgsz=self.config.image_size,
            conf=self.config.conf,
            iou=self.config.iou,
            max_det=self.config.max_det,
            augment=self.config.augment,
            agnostic_nms=self.config.agnostic_nms,
            project=self._run_path,
            name="predict",
        )

        # Process the output labels and upload them to S3
        labels_path = os.path.join(self._run_path, "predict", "labels")
        os.makedirs(output_dir, exist_ok=True)
        for file_name in os.listdir(labels_path):
            shutil.copy(
                os.path.join(labels_path, file_name),
                os.path.join(output_dir, file_name),
            )
        self.logger.info(f"Copied labels to {output_dir}")
