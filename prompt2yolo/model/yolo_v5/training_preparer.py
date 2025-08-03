import logging
import os
from typing import List, Optional

from prompt2yolo.configs import Paths, YoloV5DataConfig, YoloV5Hyperparameters
from prompt2yolo.model.utils import save_yaml
from prompt2yolo.utils.logger import setup_logger


class YoloV5Preparer:
    def __init__(
        self,
        class_names: List[str],
        paths: Paths,
        training_data_yaml_filename: str,
        hyp_yaml_filename: str,
        logger: Optional[logging.Logger] = None,
    ):
        self.paths = paths
        self.class_names = class_names
        self.training_data_yaml_filename = training_data_yaml_filename
        self.hyp_yaml_filename = hyp_yaml_filename
        self.logger = logger or setup_logger(__name__)
        self.prepare_directories()

    def prepare_directories(self):
        os.makedirs(self.paths.yolo_config_folder, exist_ok=True)
        os.makedirs(self.paths.yolo_data_folder, exist_ok=True)

    def create_yaml_files(self, hyperparams: YoloV5Hyperparameters):
        """Create dataset and hyperparameter configuration files."""

        training_data_config = YoloV5DataConfig(
            train_dir=os.path.join(self.paths.yolo_data_folder, "train/images"),
            val_dir=os.path.join(self.paths.yolo_data_folder, "val/images"),
            class_names=self.class_names,
        )

        save_yaml(
            os.path.join(
                self.paths.yolo_config_folder, self.training_data_yaml_filename
            ),
            training_data_config.to_dict(),
        )
        save_yaml(
            os.path.join(self.paths.yolo_config_folder, self.hyp_yaml_filename),
            hyperparams.to_dict(),
        )

        self.logger.info("[*] Created configuration and hyperparameter files.")
