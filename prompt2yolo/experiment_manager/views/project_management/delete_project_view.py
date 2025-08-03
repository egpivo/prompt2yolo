import boto3
import streamlit as st

from prompt2yolo.experiment_manager.models.project_model import delete_project


def render(s3: boto3.client, selected_project: str):
    with st.expander("Delete Project"):
        st.warning(
            "**This action will permanently delete the entire project and all associated data.**"
        )
        st.write(f"To confirm deletion, type the project name: **{selected_project}**")

        # Confirmation input
        confirm_project_name = st.text_input(
            "Type the project name to confirm deletion:"
        )

        if confirm_project_name == selected_project and st.button(
            f"Delete Project '{selected_project}'"
        ):
            delete_project(s3, selected_project)
            st.success(f"Project '{selected_project}' has been successfully deleted.")
            st.rerun()
