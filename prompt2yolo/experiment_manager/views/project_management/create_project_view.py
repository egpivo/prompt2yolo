import streamlit as st

from prompt2yolo.experiment_manager import PROJECT_ROOT
from prompt2yolo.experiment_manager.models.environment_model import update_env_variable
from prompt2yolo.experiment_manager.utils import is_training_running


def render(projects: list):
    st.sidebar.subheader("Define a New Project")
    project_name = st.sidebar.text_input("Enter Project Name")

    if is_training_running():
        st.sidebar.warning(
            "A training job is currently running. Please wait until it completes."
        )
        return

    if st.sidebar.button("Create Project"):
        if not project_name:
            st.sidebar.error("Please enter a valid project name.")
            return

        if project_name in projects:
            st.sidebar.error(
                f"The project '{project_name}' already exists. Please use a different name."
            )
            return

        update_env_variable(".env", "PROJECT", project_name, PROJECT_ROOT)
        projects.append(project_name)
        st.session_state["project_name"] = project_name
        st.sidebar.success(f"New project '{project_name}' created successfully!")

    if st.session_state.get("project_name"):
        st.sidebar.info(f"Current Project: {st.session_state['project_name']}")

    st.empty()
