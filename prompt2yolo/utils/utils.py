import urllib.request
import uuid
from pathlib import Path
from typing import List, Union

import boto3
import yaml

from prompt2yolo.utils.logger import setup_logger

LOGGER = setup_logger(__name__)


def download_images(images: List[str], image_path: Union[str, Path]) -> None:
    """
    Download images from a list of URLs to a local directory.

    Args:
        images (List[str]): List of image URLs to download.
        image_path (Union[str, Path]): Path to the directory where images will be saved.

    Returns:
        None
    """
    image_path = Path(image_path)
    # Create the directory if it doesn't exist
    image_path.mkdir(parents=True, exist_ok=True)

    for url in images:
        image_uuid = str(uuid.uuid4())
        img_path = image_path / f"{image_uuid}.jpg"
        urllib.request.urlretrieve(url, img_path)


def save_txt_data(txt_data: List[str], txt_path: Union[str, Path]) -> None:
    """
    Save a list of strings as separate text files in a local directory.

    Args:
        txt_data (List[str]): List of strings to save as text files.
        txt_path (Union[str, Path]): Path to the directory where text files will be saved.

    Returns:
        None
    """
    txt_path = Path(txt_path)
    # Create the directory if it doesn't exist
    txt_path.mkdir(parents=True, exist_ok=True)

    for i, txt_content in enumerate(txt_data):
        txt_uuid = str(uuid.uuid4())
        txt_file_path = txt_path / f"{txt_uuid}.txt"
        with open(txt_file_path, "w") as txt_file:
            txt_file.write(txt_content)


def load_yaml_config(file_path: str):
    """Load a YAML configuration file."""
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def list_model_ids(
    s3_client: boto3.client, bucket_name: str, project_name: str
) -> List[str]:
    """List unique model IDs under projects/{project_name}/models/."""
    prefix = f"projects/{project_name}/models/"
    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        operation_parameters = {"Bucket": bucket_name, "Prefix": prefix}
        page_iterator = paginator.paginate(**operation_parameters)

        model_ids: set = set()
        for page in page_iterator:
            for obj in page.get("Contents", []):
                key_suffix = obj["Key"][len(prefix) :]
                if "/" in key_suffix:
                    model_id = key_suffix.split("/")[0]
                    model_ids.add(model_id)

        return sorted(model_ids)
    except Exception as e:
        LOGGER.warning(f"Error exception: {e}")
        return []
