import os
import signal
from pathlib import Path

import boto3
import streamlit as st
from dotenv import load_dotenv

from prompt2yolo.utils.logger import setup_logger

# Load environment variables from .env file
load_dotenv()
LOCK_FILE_PATH = Path(".training_lock")

LOGGER = setup_logger(__name__)

# Initialize S3 client
def initialize_s3_client() -> boto3.client:
    return boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        endpoint_url=os.getenv("AWS_S3_ENDPOINT"),
    )


def create_lock_file(pid: int):
    """
    Create a file-based lock to signal that training is in progress.
    Write the PID of the training process inside it.
    """
    with LOCK_FILE_PATH.open("w") as f:
        f.write(str(pid))


def remove_lock_file():
    """Remove the file-based lock."""
    if LOCK_FILE_PATH.exists():
        LOCK_FILE_PATH.unlink()


def read_pid_from_lock() -> int:
    """Return the PID from the lock file, or -1 if none."""
    if not LOCK_FILE_PATH.exists():
        return -1
    try:
        with LOCK_FILE_PATH.open("r") as f:
            return int(f.read().strip())
    except ValueError:
        return -1


def is_pid_alive(pid: int) -> bool:
    """
    Check if a process with the given PID is running.
    This works on Unix-like systems. On Windows, you'd do it differently.
    """
    if pid <= 0:
        return False
    try:
        # Send signal 0 to check if the process is alive
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def is_training_running() -> bool:
    """
    Return True if the lock file exists and the PID inside it is still alive.
    Otherwise, remove the lock file and return False.
    """
    pid = read_pid_from_lock()
    if pid == -1:
        # No valid PID in the lock
        remove_lock_file()
        return False

    # If the PID is not alive, remove the lock
    if not is_pid_alive(pid):
        remove_lock_file()
        return False

    # Otherwise, training is still in progress
    return True


def terminate_training():
    """Terminate the training process."""
    pid = read_pid_from_lock()
    if pid > 0 and is_pid_alive(pid):
        os.kill(pid, signal.SIGTERM)
    remove_lock_file()
    st.session_state.pop("training_process", None)


def get_selected_iteration():
    return st.sidebar.slider("Select Iteration", min_value=1, max_value=10, value=1)
