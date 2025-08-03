import subprocess

import streamlit as st

from prompt2yolo.experiment_manager import PROJECT_ROOT
from prompt2yolo.experiment_manager.utils import (
    create_lock_file,
    is_training_running,
    remove_lock_file,
)
from prompt2yolo.experiment_manager.views.training.ongoing_training_view import (
    LOG_FILE_PATH,
)


def _on_process_exit_callback(process: subprocess.Popen):
    if process.poll() is not None:
        remove_lock_file()


def start_training(training_script_path, max_iterations, convergence_threshold):
    """Starts the training job and updates session state."""

    try:
        if is_training_running():
            st.warning("A training job is already running. Please wait.")
            return

        LOG_FILE_PATH.unlink(missing_ok=True)
        LOG_FILE_PATH.touch()

        with LOG_FILE_PATH.open("a") as log_file:
            process = subprocess.Popen(
                [
                    str(training_script_path),
                    "--max_iterations",
                    str(max_iterations),
                    "--convergence_threshold",
                    str(convergence_threshold),
                ],
                stdout=log_file,
                stderr=log_file,
                text=True,
                bufsize=1,
            )
            create_lock_file(process.pid)
            st.session_state["training_process"] = process
            st.session_state["training_logs"] = []
        st.success(f"Training job started with PID {process.pid}.")

    except Exception as e:
        remove_lock_file()
        st.error(f"An error occurred while starting the job: {str(e)}")


def render(selected_project: str):
    st.header(f"Start Model Training for Project: {selected_project}")

    if is_training_running():
        st.warning(
            "A training job is already running. Please go to the Ongoing Training tab to monitor progress."
        )
        return

    # Training configuration sliders
    max_iterations = st.slider("Max Iterations", min_value=1, max_value=20, value=10)
    convergence_threshold = st.slider(
        "Convergence Threshold", min_value=0.01, max_value=1.0, value=0.05
    )

    training_script_path = (
        PROJECT_ROOT / "scripts" / "automate_human_detection_model_training.sh"
    )

    if not training_script_path.exists():
        st.error(f"Training script not found at {training_script_path}")
        return

    # If the user clicks start training
    if st.button("Start Training"):
        start_training(
            training_script_path,
            max_iterations,
            convergence_threshold,
        )
