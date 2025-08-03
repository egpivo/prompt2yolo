import boto3
import streamlit as st

from prompt2yolo.configs import Paths
from prompt2yolo.experiment_manager import BUCKET_NAME
from prompt2yolo.experiment_manager.models.model_model import get_latest_iteration


def render(
    s3: boto3.client,
    selected_project: str,
):
    # Get the latest iteration
    latest_iteration = get_latest_iteration(s3, selected_project)
    if not latest_iteration:
        st.error("No iterations found for the selected project.")
        return

    paths = Paths(project=selected_project, iteration=latest_iteration)
    weights_folder = (
        f"projects/{selected_project}/models/iteration_{latest_iteration}/weights"
    )
    st.header(f"Download Trained Model (Iteration {latest_iteration})")

    # Check for best and last weights
    best_model_key = f"{weights_folder}/best.pt"
    last_model_key = f"{weights_folder}/last.pt"

    try:
        # Check if best weights exist
        with st.spinner("Checking for the best model..."):
            s3.head_object(Bucket=BUCKET_NAME, Key=best_model_key)
            st.success("Best model is available for download.")
            best_model_data = s3.get_object(Bucket=BUCKET_NAME, Key=best_model_key)[
                "Body"
            ].read()

            # Download button for the best model
            st.download_button(
                label="Download Best Model",
                data=best_model_data,
                file_name=f"{selected_project}_iteration_{latest_iteration}_best_model.pt",
                mime="application/octet-stream",
            )
    except s3.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            st.warning("Best model not found.")
        else:
            st.error("An error occurred while accessing the S3 bucket.")
            raise

    try:
        # Check if last weights exist
        with st.spinner("Checking for the last model..."):
            s3.head_object(Bucket=BUCKET_NAME, Key=last_model_key)
            st.success("Last model is available for download.")
            last_model_data = s3.get_object(Bucket=BUCKET_NAME, Key=last_model_key)[
                "Body"
            ].read()

            # Download button for the last model
            st.download_button(
                label="Download Last Model",
                data=last_model_data,
                file_name=f"{selected_project}_iteration_{latest_iteration}_last_model.pt",
                mime="application/octet-stream",
            )
    except s3.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            st.warning("Last model not found.")
        else:
            st.error("An error occurred while accessing the S3 bucket.")
            raise
