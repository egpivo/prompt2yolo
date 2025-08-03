import streamlit as st

from prompt2yolo.experiment_manager.utils import is_training_running
from prompt2yolo.experiment_manager.views.training import (
    ongoing_training_view,
    start_training_view,
)


def handle_training_action(action: str):
    """Handle actions related to training workflows."""
    if action == "Ongoing Training Job":
        _handle_ongoing_training()
    elif action == "Start Training":
        _handle_start_training()


def _handle_ongoing_training():
    """Render the ongoing training view."""
    if is_training_running():
        ongoing_training_view.render()
    else:
        st.info("No training job is currently running.")


def _handle_start_training():
    """Validate and render the start training view."""
    if "project_name" not in st.session_state:
        st.warning("No project selected. Please create or load a project first.")
        return

    start_training_view.render(selected_project=st.session_state["project_name"])
