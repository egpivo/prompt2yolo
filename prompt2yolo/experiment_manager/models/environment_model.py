import os

from dotenv import dotenv_values, set_key

from prompt2yolo.utils.logger import setup_logger

LOGGER = setup_logger(__name__)


def update_env_variable(env_file: str, key: str, value: str, base_dir: str = None):
    try:
        # If a base directory is provided, construct the full path
        if base_dir:
            env_file = os.path.join(base_dir, env_file)

        # Load existing environment variables
        env_config = dotenv_values(env_file)

        # Check if the key exists
        if key in env_config:
            LOGGER.info(f"Updating {key} in {env_file}...")
        else:
            LOGGER.info(f"Adding {key} to {env_file}...")

        set_key(env_file, key, value)
        LOGGER.info(f"{key} updated successfully to: {value}")
    except Exception as e:
        LOGGER.info(f"Error updating {key} in {env_file}: {e}")
