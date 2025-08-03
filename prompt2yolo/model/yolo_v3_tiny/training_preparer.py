import logging
import os
from shutil import rmtree
from typing import Optional

from prompt2yolo.configs import (
    Paths,
    YoloModelArchitecture,
    YoloV3TinyDataConfig,
    YoloV3TinyHyperparameters,
)
from prompt2yolo.utils.logger import setup_logger


class YoloV3TinyPreparer:
    """
    Prepares the necessary data and configuration files for training a YOLOv3-tiny model.

    This class handles directory creation, copying data, generating the train.txt file,
    and creating the necessary YOLO configuration files for training.

    Caveat:
        This class requires a YOLOv3-tiny configuration template file (`yolov3-tiny.cfg`)
        which is typically obtained from the Darknet repository. Make sure to clone the
        Darknet repository from https://github.com/AlexeyAB/darknet and reference the
        appropriate path to `yolov3-tiny.cfg` within the cloned repository.
    """

    def __init__(
        self,
        class_names: list,
        paths: Paths,
        logger: Optional[logging.Logger] = None,
    ):
        self.paths = paths
        self.class_names = class_names
        self.logger = logger or setup_logger(__name__)

    def prepare_directories(self):
        """Ensure train directories exist and clear previous contents if needed."""
        os.makedirs(self.paths.yolo_data_folder, exist_ok=True)
        os.makedirs(self.paths.yolo_config_folder, exist_ok=True)

        for subdir in ["train"]:
            for folder_type in ["images", "labels"]:
                directory = os.path.join(
                    self.paths.yolo_data_folder, subdir, folder_type
                )
                if os.path.exists(directory):
                    rmtree(directory)
                os.makedirs(directory, exist_ok=True)
        self.logger.info("[*] Train directories are ready.")

    def create_train_txt(self, train_txt_path: str):
        """Create train.txt file listing image paths."""
        image_dir = os.path.join(self.paths.yolo_data_folder, "train/images")
        with open(train_txt_path, "w") as file:
            for img_name in os.listdir(image_dir):
                if img_name.endswith((".jpg", ".jpeg", ".png")):
                    img_path = os.path.join(image_dir, img_name)
                    file.write(img_path + "\n")
        self.logger.info(f"[*] Created 'train.txt' at {train_txt_path}.")

    def create_obj_files(
        self,
        num_of_classes: int,
        obj_names_path: str,
        obj_data_path: str,
        train_txt_path: str,
        save_model_path: str,
    ):
        """Create obj.names and obj.data files."""
        # Create obj.names
        with open(obj_names_path, "w") as file:
            for class_name in self.class_names:
                file.write(class_name + "\n")
        self.logger.info(f"[*] Created 'obj.names' at {obj_names_path}.")

        YoloV3TinyDataConfig(
            train_dir=train_txt_path,
            class_name_file=obj_names_path,
            num_of_classes=num_of_classes,
            save_model_path=save_model_path,
        ).save_to_obj_data(obj_data_path)

        self.logger.info(f"[*] Created 'obj.data' at {obj_data_path}.")

    def update_cfg_file(
        self,
        cfg_template_path: str,
        output_cfg_path: str,
        num_classes: int,
        hyperparameters: YoloV3TinyHyperparameters,
        architecture: YoloModelArchitecture,
    ):
        """
        Update the YOLOv3-tiny.cfg file with specified parameters using a hyperparameter class
        and save the data configuration file.
        """
        indexes_of_yolo_preceding_layers = architecture.layer_indexes
        filters = (num_classes + hyperparameters.filters_base) * 3

        # Read the cfg template file in darknet/cfg/
        with open(cfg_template_path, "r") as file:
            file_content = file.readlines()

        # Update specific lines based on keywords
        for i, line in enumerate(file_content):
            if line.startswith("batch="):
                file_content[i] = f"batch={hyperparameters.batch_size}\n"
            elif line.startswith("max_batches"):
                file_content[i] = f"max_batches={hyperparameters.max_batches}\n"
            elif line.startswith("steps"):
                file_content[i] = f"steps={','.join(map(str, hyperparameters.steps))}\n"
            elif line.startswith("learning_rate="):
                file_content[i] = f"learning_rate={hyperparameters.learning_rate}\n"
            elif line.strip().startswith("classes="):
                file_content[i] = f"classes={num_classes}\n"
            elif i in indexes_of_yolo_preceding_layers:
                file_content[i] = f"filters={filters}\n"

        # Ensure output directory exists "should be darknet/cfg/..."
        with open(output_cfg_path, "w") as file:
            file.writelines(file_content)

        self.logger.info(f"[*] Created '{output_cfg_path}' configuration file.")
