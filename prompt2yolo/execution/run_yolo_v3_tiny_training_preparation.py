import argparse
import os

import yaml
from dotenv import load_dotenv

from prompt2yolo.data.utils import clean_directory
from prompt2yolo.model.yolo_v3_tiny.training_preparer import YoloV3TinyPreparer
from prompt2yolo.utils.logger import setup_logger
from prompt2yolo.utils.s3_data_downloader import S3DataDownloader

load_dotenv()

from prompt2yolo.configs import Paths, YoloModelArchitecture, YoloV3TinyHyperparameters


def fetch_args() -> argparse.Namespace:
    """
    Parse command-line arguments for data preparation.
    """
    parser = argparse.ArgumentParser(description="YOLOv3 Data Preparation Script")

    parser.add_argument(
        "--input_yaml",
        type=str,
        required=True,
        help="Path to the input YAML file containing class and configuration details.",
    )
    parser.add_argument(
        "--obj_names_filename",
        type=str,
        required=True,
        help="Filename to save the object class names.",
    )
    parser.add_argument(
        "--obj_data_filename",
        type=str,
        required=True,
        help="Filename to save the obj.data configuration.",
    )
    parser.add_argument(
        "--train_txt_filename",
        type=str,
        required=True,
        help="Filename to save the list of training images.",
    )
    parser.add_argument(
        "--save_model_path",
        type=str,
        required=True,
        help="Path to save the trained model weights.",
    )
    parser.add_argument(
        "--cfg_template_path",
        type=str,
        required=True,
        help="Path to the YOLOv3-tiny.cfg template file.",
    )
    parser.add_argument(
        "--output_cfg_path",
        type=str,
        required=True,
        help="Path to save the updated YOLOv3-tiny configuration file.",
    )
    parser.add_argument(
        "--does_download_data_from_s3",
        action="store_true",
        help="Flag to indicate whether images and labels should be downloaded from S3.",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    """
    Main function to run data preparation and configuration creation.
    """
    logger = setup_logger()
    with open(args.input_yaml, "r") as file:
        config = yaml.safe_load(file)

    classes = config.get("classes")
    paths = Paths(local_data_path=os.getenv("LOCAL_DATA_PATH"))

    data_preparer = YoloV3TinyPreparer(
        class_names=classes,
        paths=paths,
        logger=logger,
    )

    if args.does_download_data_from_s3:
        clean_directory(paths.image_folder, logger)
        clean_directory(paths.label_folder, logger)
        S3DataDownloader(
            s3_image_folder=paths.s3_image_folder,
            s3_label_folder=paths.s3_label_folder,
            image_folder=paths.image_folder,
            label_folder=paths.label_folder,
            logger=logger,
        ).download_images_and_labels()

    # Copy data and create train.txt
    data_preparer.create_train_txt(train_txt_path=args.train_txt_filename)

    # Create obj.names and obj.data files
    data_preparer.create_obj_files(
        num_of_classes=len(classes),
        obj_names_path=args.obj_names_filename,
        obj_data_path=args.obj_data_filename,
        train_txt_path=args.train_txt_filename,
        save_model_path=args.save_model_path,
    )

    hyperparameters = YoloV3TinyHyperparameters()
    data_preparer.update_cfg_file(
        cfg_template_path=args.cfg_template_path,
        output_cfg_path=args.output_cfg_path,
        num_classes=len(classes),
        hyperparameters=hyperparameters,
        architecture=YoloModelArchitecture.YOLOV3_TINY,
    )


if __name__ == "__main__":
    run(fetch_args())
