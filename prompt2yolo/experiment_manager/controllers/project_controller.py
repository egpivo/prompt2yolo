import streamlit as st

from prompt2yolo.experiment_manager.controllers.dataset_controller import (
    handle_dataset_action,
)
from prompt2yolo.experiment_manager.controllers.evaluation_controller import (
    handle_evaluation_action,
)
from prompt2yolo.experiment_manager.views.project_management import (
    create_project_view,
    delete_project_view,
    download_model_view,
    prompt_setup_view,
)
from prompt2yolo.experiment_manager.views.training import start_training_view


def handle_project_action(s3, action, projects):
    """Handle actions related to project workflows."""
    if action == "Create New Project":
        _create_new_project(projects)
    elif action == "Check Existing Project":
        _check_existing_project(s3, projects)


def _create_new_project(projects):
    """Render the Create New Project workflow."""
    create_project_view.render(projects)
    if st.session_state.get("project_name"):
        workflow_action = st.sidebar.radio(
            "Select Workflow Step",
            ["Setup Prompts", "Start Model Training"],
            key="workflow_action",
        )

        if workflow_action == "Setup Prompts":
            st.title("Setup Prompts for Your Project")
            prompt_setup_view.render()
        elif workflow_action == "Start Model Training":
            st.title("Start Model Training")
            _validate_and_start_training()


def _validate_and_start_training():
    """Validate prompts and start model training."""
    prompts = st.session_state.get("prompt_data", {}).get("prompts", [])
    if len(prompts) < 1:
        st.warning(
            "You need to set up at least one prompt before starting model training."
        )
        st.info("Go to 'Setup Prompts' to add prompts.")
    else:
        start_training_view.render(selected_project=st.session_state["project_name"])


def _check_existing_project(s3, projects):
    """Handle loading and managing existing projects."""
    if not projects:
        st.sidebar.error("No existing projects found.")
        return

    selected_project = st.sidebar.selectbox(
        "Select Existing Project:", options=projects
    )
    if st.sidebar.button("Load Project"):
        st.session_state["project_name"] = selected_project
        st.success(f"Loaded project: {selected_project}")

    if st.session_state.get("project_name"):
        _render_existing_project_workflow(s3)


def _render_existing_project_workflow(s3):
    """Render workflows for an existing project."""
    section = st.sidebar.radio(
        "Project Actions",
        [
            "Dataset Overview",
            "Evaluation Workflow",
            "Model Management",
            "Administrative Tools",
        ],
    )

    if section == "Dataset Overview":
        handle_dataset_action(s3)
    elif section == "Evaluation Workflow":
        handle_evaluation_action(s3)
    elif section == "Model Management":
        download_model_view.render(
            s3,
            st.session_state["project_name"],
        )
    elif section == "Administrative Tools":
        delete_project_view.render(
            s3,
            st.session_state["project_name"],
        )
