import argparse
import os

import boto3
import yaml
from dotenv import load_dotenv

from prompt2yolo.configs import Paths
from prompt2yolo.data.utils import clean_directory
from prompt2yolo.model.yolo_v5.inference_preparer import YoloV5Preparer
from prompt2yolo.utils.logger import setup_logger
from prompt2yolo.utils.s3_data_downloader import S3DataDownloader, S3ModelDownloader

load_dotenv()


def fetch_args() -> argparse.Namespace:
    """
    Parse command-line arguments for data preparation.
    """
    parser = argparse.ArgumentParser(description="YOLO Data Preparation Script")
    parser.add_argument("--input_yaml", required=True, help="Path to input YAML file.")
    parser.add_argument(
        "--testing_data_yaml_filename",
        required=True,
        help="Filename for testing data configuration.",
    )
    parser.add_argument(
        "--does_download_data_from_s3",
        action="store_true",
        help="Download images and labels from S3.",
    )
    parser.add_argument(
        "--does_download_model_from_s3",
        action="store_true",
        help="Download models from S3.",
    )
    parser.add_argument(
        "--model_folder",
        type=str,
        default=None,
        help="Model local folder to download.",
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
    # Note: we only use the initial test examples
    paths = Paths(
        local_data_path=os.getenv("LOCAL_DATA_PATH"),
        mode="test",
        iteration=args.iteration,
    )
    data_preparer = YoloV5Preparer(
        class_names=classes,
        paths=paths,
        testing_data_yaml_filename=args.testing_data_yaml_filename,
        logger=logger,
    )
    s3_client = boto3.client("s3", endpoint_url=os.getenv("AWS_S3_ENDPOINT"))
    os.getenv("AWS_S3_BUCKET_NAME")
    os.getenv("PROJECT")

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
        data_preparer.prepare_test_data()
    elif args.iteration > 1:
        data_preparer.copy_from_initial_iteration()

    if args.does_download_model_from_s3:
        S3ModelDownloader(
            s3_model_folder=f"{paths.s3_model_weights_folder}",
            model_folder=paths.yolo_model_folder,
            logger=logger,
        ).download_model_files()

    data_preparer.create_yaml_files()


if __name__ == "__main__":
    run(fetch_args())
