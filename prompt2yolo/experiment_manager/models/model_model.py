import boto3
import streamlit as st
from botocore.exceptions import ClientError

from prompt2yolo.experiment_manager import BUCKET_NAME
from prompt2yolo.utils.logger import setup_logger

LOGGER = setup_logger(__name__)


def delete_model(s3_client, project_name, model_id, result_type="model"):
    """Delete all objects related to a specific model ID in the specified project."""
    try:
        # Use paginator to list all objects
        paginator = s3_client.get_paginator("list_objects_v2")
        model_prefix = f"projects/{project_name}/{result_type}/{model_id}/"
        operation_parameters = {"Bucket": BUCKET_NAME, "Prefix": model_prefix}
        page_iterator = paginator.paginate(**operation_parameters)

        objects_to_delete = []
        for page in page_iterator:
            objects = page.get("Contents", [])
            for obj in objects:
                objects_to_delete.append({"Key": obj["Key"]})

        if objects_to_delete:
            # Delete objects in batches of 1000
            for i in range(0, len(objects_to_delete), 1000):
                batch = objects_to_delete[i : i + 1000]
                s3_client.delete_objects(Bucket=BUCKET_NAME, Delete={"Objects": batch})

            st.success(f"Model '{model_id}' has been deleted successfully.")
        else:
            st.warning(f"No objects found for model '{model_id}'.")
    except ClientError as e:
        st.error(f"Failed to delete model '{model_id}': {e}")


def get_latest_iteration(s3: boto3.client, project: str) -> int:
    """Get the latest iteration number from the S3 bucket."""
    prefix = f"projects/{project}/models/"
    response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix, Delimiter="/")

    if "CommonPrefixes" not in response:
        LOGGER.warning("No CommonPrefixes found in the response.")

    iterations = [
        int(folder["Prefix"].split("_")[-1].strip("/"))
        for folder in response.get("CommonPrefixes", [])
        if folder["Prefix"].split("_")[-1].strip("/").isdigit()
    ]
    return max(iterations) if iterations else 1
