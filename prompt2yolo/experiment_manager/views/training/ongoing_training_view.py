from pathlib import Path

import streamlit as st
from ansi2html import Ansi2HTMLConverter
from filelock import FileLock
from streamlit_autorefresh import st_autorefresh

from prompt2yolo.experiment_manager.utils import (
    is_training_running,
    read_pid_from_lock,
    terminate_training,
)

LOG_FILE_PATH = Path("training_logs.txt")
LOCK_FILE_PATH = LOG_FILE_PATH.with_suffix(".lock")


def load_logs():
    """Load logs from the file into session state with file locking."""
    if LOG_FILE_PATH.exists():
        with FileLock(LOCK_FILE_PATH):
            with LOG_FILE_PATH.open("r") as log_file:
                st.session_state["training_logs"] = log_file.readlines()
    else:
        st.session_state["training_logs"] = []


def save_logs():
    """Save logs from session state to the file with file locking."""
    with FileLock(LOCK_FILE_PATH):
        with LOG_FILE_PATH.open("w") as log_file:
            log_file.writelines(st.session_state["training_logs"])


def fetch_logs():
    """Fetch logs from the running process and update the session state."""
    process = st.session_state["training_process"]
    if process and process.poll() is None:
        try:
            # Read a line from stdout
            output = process.stdout.readline().strip()
            if output:
                st.session_state["training_logs"].append(output + "\n")

            # Read a line from stderr
            error = process.stderr.readline().strip()
            if error:
                st.session_state["training_logs"].append(error + "\n")

            # Save logs to file after every update
            save_logs()
        except Exception as e:
            st.error(f"Failed to fetch logs: {str(e)}")


def display_logs(log_container):
    """Display the last 20 logs with colors."""
    logs = st.session_state.get("training_logs", [])
    conv = Ansi2HTMLConverter()
    html_logs = "\n".join([conv.convert(log.strip(), full=False) for log in logs[-20:]])
    log_container.markdown(html_logs, unsafe_allow_html=True)


def render():
    """Render the Ongoing Training view **without** blocking the Streamlit script."""
    st.header("Ongoing Training Job")

    # Optionally auto-refresh the page every 3 seconds
    # so logs update in "real-time" without blocking.
    st_autorefresh(interval=3000)

    # Ensure we have a list for logs in session state
    if "training_logs" not in st.session_state:
        st.session_state["training_logs"] = []

    # Load logs from file to restore them if the page was refreshed
    load_logs()
    if is_training_running():
        pid = read_pid_from_lock()
        st.success(f"A training job is running (PID: {pid}).")
        terminate_job_button = st.button("Terminate Job")
        if terminate_job_button:
            terminate_training()
            st.error("Training job terminated.")
        else:
            log_container = st.empty()
            with LOG_FILE_PATH.open("r") as log_file:
                logs = log_file.readlines()
                conv = Ansi2HTMLConverter()
                html_logs = "\n".join(
                    [conv.convert(log.strip(), full=False) for log in logs[-20:]]
                )
                log_container.markdown(html_logs, unsafe_allow_html=True)

    else:
        st.info("No training job is currently running.")
