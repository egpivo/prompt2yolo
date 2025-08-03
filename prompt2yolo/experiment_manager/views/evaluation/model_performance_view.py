import os
from io import BytesIO

import boto3
import streamlit as st
from PIL import Image

from prompt2yolo.configs import Paths
from prompt2yolo.experiment_manager.models.image_model import (
    list_s3_images,
    load_image_from_s3,
)


def render(
    s3: boto3.client,
    selected_project: str,
    iteration: int = 1,
):
    paths = Paths(project=selected_project, iteration=iteration)
    st.header("Model Performance")

    data_option = st.radio(
        "Select the type of results to display:",
        options=["Training", "Inference"],
        index=0,
    )

    view_option = st.radio(
        "Select the type of results to display:",
        options=["Performance Metrics", "Prediction Results"],
        index=0,
    )

    # Map selection to corresponding S3 folder
    if view_option == "Performance Metrics" and data_option == "Training":
        folder = paths.s3_model_training_performance_folder
    elif view_option == "Performance Metrics" and data_option == "Inference":
        folder = paths.s3_model_inference_performance_folder
    elif view_option == "Prediction Results" and data_option == "Training":
        folder = paths.s3_model_training_prediction_folder
    else:
        folder = paths.s3_model_inference_prediction_folder
    # Load and display images from the selected folder
    image_keys = list_s3_images(s3, folder)

    if not image_keys:
        st.write(f"No {view_option.lower()} found.")
    else:
        cols = st.columns(3)
        for idx, img_key in enumerate(image_keys):
            img_bytes = load_image_from_s3(s3, img_key)
            image = Image.open(BytesIO(img_bytes))
            with cols[idx % 3]:
                st.image(
                    image,
                    caption=os.path.basename(img_key),
                    use_column_width=True,
                )
