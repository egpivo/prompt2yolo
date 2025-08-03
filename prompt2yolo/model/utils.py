import yaml
from botocore.client import BaseClient

from prompt2yolo.utils.utils import list_model_ids


def get_latest_model_version(
    s3_client: BaseClient, bucket_name: str, project_name: str
) -> str:
    try:
        # Use the existing list_model_ids function to fetch sorted model IDs
        model_ids = list_model_ids(s3_client, bucket_name, project_name)
        if not model_ids:
            raise ValueError(
                f"No model IDs found under 'projects/{project_name}/models/'."
            )

        # The last item in the sorted list is the latest model version
        latest_model_id = model_ids[-1]
        return latest_model_id
    except Exception as e:
        raise RuntimeError(f"Failed to fetch the latest model version: {e}")


def save_yaml(file_path: str, data: dict):
    """Save data to a YAML file."""
    with open(file_path, "w") as file:
        yaml.dump(data, file, default_flow_style=False)
