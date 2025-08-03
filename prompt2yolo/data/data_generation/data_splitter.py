import logging
import os
from shutil import copyfile, rmtree
from typing import Optional

from sklearn.model_selection import train_test_split

from prompt2yolo.configs import Paths
from prompt2yolo.utils.logger import setup_logger


class DataSplitter:
    """Splits data into train, val, and test sets and organizes into YOLO format."""

    def __init__(
        self,
        paths: Paths,
        val_ratio: float,
        test_ratio: float,
        logger: Optional[logging.Logger] = None,
    ):
        self.paths = paths
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.logger = logger or setup_logger(__name__)
        self._prepare_directories()

    def _prepare_directories(self):
        """Prepare and clear directories for train/val/test splits."""
        for split in ["train", "val", "test"]:
            for folder in ["images", "labels"]:
                dir_path = os.path.join(self.paths.yolo_data_folder, split, folder)
                rmtree(dir_path, ignore_errors=True)
                os.makedirs(dir_path, exist_ok=True)
        self.logger.info("[*] Directories prepared for splits.")

    def _copy_files(self, files, split):
        """Copy images and labels to the specified split."""
        for file in files:
            copyfile(
                os.path.join(self.paths.image_folder, file),
                os.path.join(self.paths.yolo_data_folder, f"{split}/images", file),
            )
            label = file.replace(".jpg", ".txt").replace(".png", ".txt")
            label_src = os.path.join(self.paths.label_folder, label)
            label_dst = os.path.join(
                self.paths.yolo_data_folder, f"{split}/labels", label
            )
            if os.path.exists(label_src):
                copyfile(label_src, label_dst)

    def split(self):
        """Split data and organize files into train, val, and test directories."""
        images = [
            f
            for f in os.listdir(self.paths.image_folder)
            if f.endswith((".jpg", ".png"))
        ]
        if 0 < self.test_ratio < 1:
            train_val, test = train_test_split(images, test_size=self.test_ratio)
        elif self.test_ratio == 1:
            train_val, test = [], images
        elif self.test_ratio == 0:
            train_val, test = images, []
        else:
            raise ValueError(
                f"Enter correct test ratio in [0, 1], but got {self.test_ratio}"
            )

        if train_val:
            if 0 < self.val_ratio < 1:
                train, val = train_test_split(train_val, test_size=self.val_ratio)
            elif self.val_ratio == 0:
                train, val = images, []
            else:
                raise ValueError(
                    f"Enter correct test ratio in [0, 1), but got {self.val_ratio}"
                )
        else:
            train, val = [], []

        for split, files in zip(["train", "val", "test"], [train, val, test]):
            self._copy_files(files, split)

        self.logger.info(
            f"[*] Data split: {len(train)} train, {len(val)} val, {len(test)} test."
        )
