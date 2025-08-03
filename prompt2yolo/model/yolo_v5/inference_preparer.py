import logging
import os
import shutil
from shutil import copyfile
from typing import List, Optional

from prompt2yolo.configs import Paths, YoloV5DataConfig
from prompt2yolo.model.utils import save_yaml
from prompt2yolo.utils.logger import setup_logger


class YoloV5Preparer:
    def __init__(
        self,
        class_names: List[str],
        paths: Paths,
        testing_data_yaml_filename: str,
        logger: Optional[logging.Logger] = None,
    ):
        self.paths = paths
        self.class_names = class_names
        self.testing_data_yaml_filename = testing_data_yaml_filename
        self.logger = logger or setup_logger(__name__)

    def prepare_test_data(self):
        """Copy all images and corresponding labels to the test directory. Only used when we download data from S3"""
        image_list = [
            img
            for img in os.listdir(self.paths.image_folder)
            if img.endswith((".jpg", ".png"))
        ]

        self._copy_files(image_list, "test")
        self.logger.info(f"[*] Prepared test data: {len(image_list)} images")

    def _copy_files(self, file_list: List[str], split_type: str):
        """Copy image and corresponding label files to the target directory."""
        for file in file_list:
            src_image = os.path.join(self.paths.image_folder, file)
            dst_image = os.path.join(
                self.paths.yolo_data_folder, f"{split_type}/images", file
            )
            copyfile(src_image, dst_image)

            # Copy corresponding label file if it exists
            label_file = os.path.splitext(file)[0] + ".txt"
            src_label = os.path.join(self.paths.label_folder, label_file)
            dst_label = os.path.join(
                self.paths.yolo_data_folder, f"{split_type}/labels", label_file
            )
            if os.path.exists(src_label):
                copyfile(src_label, dst_label)

    def copy_from_initial_iteration(self):
        """Copy test data from the initial iteration."""
        for subdir in ["test/images", "test/labels"]:
            src_dir = os.path.join(self.paths.yolo_data_init_folder, subdir)
            tgt_dir = os.path.join(self.paths.yolo_data_folder, subdir)
            os.makedirs(tgt_dir, exist_ok=True)

            for file_name in os.listdir(src_dir):
                src_path = os.path.join(src_dir, file_name)
                tgt_path = os.path.join(tgt_dir, file_name)
                if os.path.isfile(src_path):
                    shutil.copy(src_path, tgt_path)
        self.logger.info(
            f"[*] Copied test data from '{self.paths.yolo_data_init_folder}' to '{self.paths.yolo_data_folder}'."
        )

    def create_yaml_files(self):
        """Create dataset configuration YAML file."""
        testing_data_config = YoloV5DataConfig(
            train_dir=os.path.join(self.paths.yolo_data_folder, "train/images"),
            val_dir=os.path.join(self.paths.yolo_data_folder, "test/images"),
            class_names=self.class_names,
        )

        yaml_path = os.path.join(
            self.paths.yolo_config_folder, self.testing_data_yaml_filename
        )
        save_yaml(yaml_path, testing_data_config.to_dict())

        self.logger.info(f"[*] Created data configuration file: {yaml_path}")
