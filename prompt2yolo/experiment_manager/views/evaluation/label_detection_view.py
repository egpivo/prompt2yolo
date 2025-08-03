import os
from io import BytesIO

import boto3
import streamlit as st
from PIL import Image

from prompt2yolo.configs import Paths
from prompt2yolo.experiment_manager import BUCKET_NAME
from prompt2yolo.experiment_manager.models.image_model import (
    list_s3_images,
    load_image_from_s3,
)
from prompt2yolo.utils.utils import list_model_ids


def render(
    s3: boto3.client,
    selected_project: str,
    iteration: int = 1,
):
    paths = Paths(project=selected_project, iteration=iteration)
    st.header("Evaluation Results")

    # Folder selection
    view_option = st.selectbox(
        "Select View",
        ["Category Results", "Statistics"],
        index=0,
    )

    if view_option == "Category Results":
        category = st.selectbox(
            "Select Category",
            ["False Positive", "False Negative", "True Positive", "True Negative"],
            index=0,
        )
        category_input = category.replace(" ", "_").lower()

        evaluation_model_ids = list_model_ids(s3, BUCKET_NAME, selected_project)

        if not evaluation_model_ids:
            st.error("No models found for evaluation results.")
        else:
            # Update the path based on the selected category
            category_path = (
                f"{paths.s3_label_detection_category_folder}/{category_input}"
            )
            evaluation_image_keys = list_s3_images(s3, category_path)

            if not evaluation_image_keys:
                st.write(f"No evaluation results found for {category}.")
            else:
                cols = st.columns(3)
                for idx, img_key in enumerate(evaluation_image_keys):
                    img_bytes = load_image_from_s3(s3, img_key)
                    image = Image.open(BytesIO(img_bytes))
                    with cols[idx % 3]:
                        st.image(
                            image,
                            caption=os.path.basename(img_key),
                            use_column_width=True,
                        )

    elif view_option == "Statistics":
        statistics_path = paths.s3_label_detection_statistics_folder
        statistics_keys = list_s3_images(s3, statistics_path)

        if not statistics_keys:
            st.write(f"No evaluation results found in {statistics_path}.")
        else:
            cols = st.columns(3)
            for idx, img_key in enumerate(statistics_keys):
                img_bytes = load_image_from_s3(s3, img_key)
                image = Image.open(BytesIO(img_bytes))
                with cols[idx % 3]:
                    st.image(
                        image, caption=os.path.basename(img_key), use_column_width=True
                    )
