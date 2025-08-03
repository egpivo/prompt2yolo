import argparse
import os

import yaml
from dotenv import load_dotenv

from prompt2yolo.configs import Paths, YoloV5Hyperparameters
from prompt2yolo.data.utils import clean_directory
from prompt2yolo.model.yolo_v5.training_preparer import YoloV5Preparer
from prompt2yolo.utils.logger import setup_logger
from prompt2yolo.utils.s3_data_downloader import S3DataDownloader

load_dotenv()


def fetch_args() -> argparse.Namespace:
    """
    Parse command-line arguments for data preparation.
    """
    parser = argparse.ArgumentParser(description="YOLO Data Preparation Script")
    parser.add_argument("--input_yaml", required=True, help="Path to input YAML file.")
    parser.add_argument(
        "--training_data_yaml_filename",
        required=True,
        help="Filename for training data configuration.",
    )
    parser.add_argument(
        "--hyp_config_name",
        required=True,
        help="Filename for hyperparameter configuration.",
    )
    parser.add_argument(
        "--does_download_data_from_s3",
        action="store_true",
        help="Download images and labels from S3.",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=1,
        help="Iteration number for the pipeline. Defaults to 1.",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    """
    Main function to run data preparation and configuration creation.

    [TODO]: add an option to control the number of downloading images
    """
    logger = setup_logger()

    with open(args.input_yaml, "r") as file:
        config = yaml.safe_load(file)

    classes = config.get("classes")

    paths = Paths(
        local_data_path=os.getenv("LOCAL_DATA_PATH"),
        mode="train",
        iteration=args.iteration,
    )

    data_preparer = YoloV5Preparer(
        class_names=classes,
        paths=paths,
        training_data_yaml_filename=args.training_data_yaml_filename,
        hyp_yaml_filename=args.hyp_config_name,
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

    data_preparer.create_yaml_files(YoloV5Hyperparameters())


if __name__ == "__main__":
    run(fetch_args())
