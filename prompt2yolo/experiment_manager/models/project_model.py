from typing import Dict, List

import boto3
import streamlit as st
from botocore.exceptions import ClientError, NoCredentialsError

from prompt2yolo.experiment_manager import BUCKET_NAME
from prompt2yolo.utils.logger import setup_logger

LOGGER = setup_logger(__name__)


def list_projects(s3_client: boto3.client, prefix: str = "projects/") -> List[str]:
    try:
        response = s3_client.list_objects_v2(
            Bucket=BUCKET_NAME, Prefix=prefix, Delimiter="/"
        )
        return [
            prefix.get("Prefix").split("/")[1]
            for prefix in response.get("CommonPrefixes", [])
        ]
    except NoCredentialsError:
        st.error("AWS credentials not available.")
        return []
    except Exception as e:
        st.error(f"An error occurred while listing projects: {e}")
        return []


# Delete a project in the S3 bucket
def delete_project(s3_client: boto3.client, project_name: str) -> None:
    """Delete all objects in the specified project folder in S3."""
    try:
        project_prefix = f"projects/{project_name}/"
        paginator = s3_client.get_paginator("list_objects_v2")
        operation_parameters = {"Bucket": BUCKET_NAME, "Prefix": project_prefix}
        page_iterator = paginator.paginate(**operation_parameters)

        objects_to_delete: List[Dict[str, str]] = []
        for page in page_iterator:
            objects = page.get("Contents", [])
            for obj in objects:
                key = obj.get("Key")
                if key:
                    objects_to_delete.append({"Key": key})
                else:
                    LOGGER.warning(
                        f"Skipping malformed or empty key in project '{project_name}'."
                    )

        if not objects_to_delete:
            st.warning(
                f"No objects found in project '{project_name}'. Nothing to delete."
            )
            return

        # Delete objects in batches of 1000 (S3 API limit)
        for i in range(0, len(objects_to_delete), 1000):
            batch = objects_to_delete[i : i + 1000]
            try:
                response = s3_client.delete_objects(
                    Bucket=BUCKET_NAME, Delete={"Objects": batch}
                )
                LOGGER.info(f"Deleted batch {i // 1000 + 1}: {len(batch)} objects.")
                if "Errors" in response:
                    for error in response["Errors"]:
                        LOGGER.error(
                            f"Error deleting {error['Key']}: {error['Message']}"
                        )
            except ClientError as e:
                LOGGER.error(f"Failed to delete batch {i // 1000 + 1}: {e}")
                st.error(
                    f"Failed to delete batch {i // 1000 + 1} for project '{project_name}': {e}"
                )
                return

        st.success(f"Project '{project_name}' has been deleted successfully.")
    except ClientError as e:
        LOGGER.error(f"Failed to delete project '{project_name}': {e}")
        st.error(f"Failed to delete project '{project_name}': {e}")
    except Exception as e:
        LOGGER.error(f"Unexpected error while deleting project '{project_name}': {e}")
        st.error(f"Unexpected error occurred: {e}")
