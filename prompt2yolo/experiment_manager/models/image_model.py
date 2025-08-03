from typing import List

import boto3
import streamlit as st
from botocore.exceptions import NoCredentialsError

from prompt2yolo.experiment_manager import BUCKET_NAME
from prompt2yolo.utils.logger import setup_logger

LOGGER = setup_logger(__name__)


def list_s3_images(s3_client: boto3.client, prefix: str = "") -> List[str]:
    """List images in an S3 bucket, handling pagination."""
    try:
        image_keys: List[str] = []
        paginator = s3_client.get_paginator("list_objects_v2")
        operation_parameters = {"Bucket": BUCKET_NAME, "Prefix": prefix}

        page_iterator = paginator.paginate(**operation_parameters)
        for page in page_iterator:
            contents = page.get("Contents", [])
            for content in contents:
                key = content["Key"]
                if key.lower().endswith(("png", "jpg", "jpeg")):
                    image_keys.append(key)
        return image_keys
    except NoCredentialsError:
        st.error("AWS credentials not available.")
        return []


# Load an image from S3
def load_image_from_s3(s3_client: boto3.client, key: str) -> bytes:
    """Load image from S3."""
    response = s3_client.get_object(Bucket=BUCKET_NAME, Key=key)
    return response["Body"].read()


# Count images in a specific S3 folder
def count_images_in_s3(s3_client: boto3.client, prefix: str = "") -> int:
    """Count the number of images in an S3 bucket folder, handling pagination."""
    try:
        image_count = 0
        paginator = s3_client.get_paginator("list_objects_v2")
        operation_parameters = {"Bucket": BUCKET_NAME, "Prefix": prefix}
        page_iterator = paginator.paginate(**operation_parameters)

        for page in page_iterator:
            contents = page.get("Contents", [])
            image_files = [
                content["Key"]
                for content in contents
                if content["Key"].lower().endswith(("png", "jpg", "jpeg"))
            ]
            image_count += len(image_files)

        return image_count
    except NoCredentialsError:
        st.error("AWS credentials not available.")
        return 0


# Paginate image keys
def paginate_images(
    image_keys: List[str], page_size: int, current_page: int
) -> List[str]:
    """Paginate the list of image keys."""
    start_index = current_page * page_size
    end_index = start_index + page_size
    return image_keys[start_index:end_index]
