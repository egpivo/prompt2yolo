import os
from io import BytesIO

import boto3
import streamlit as st
from PIL import Image

from prompt2yolo.configs import Paths
from prompt2yolo.experiment_manager.models.image_model import (
    list_s3_images,
    load_image_from_s3,
    paginate_images,
)


def render(
    s3: boto3.client,
    selected_project: str,
    iteration: int = 1,
):
    paths = Paths(project=selected_project, iteration=iteration)

    st.header(f"Dataset for Project: {selected_project}")
    mode = st.radio(
        "Select Dataset to Display:",
        options=["train", "val", "test"],
        format_func=lambda x: x.capitalize(),
    )

    paths.mode = mode
    image_keys = list_s3_images(s3, prefix=paths.s3_image_folder)

    if not image_keys:
        st.write(f"No images found in {mode} dataset.")
    else:
        page_size = st.slider(
            f"Images per page ({mode}):", min_value=1, max_value=20, value=5
        )
        total_pages = (len(image_keys) + page_size - 1) // page_size
        current_page = st.session_state.get(f"{mode}_current_page", 0)

        selected_page = st.selectbox(
            f"Select Page ({mode}):",
            options=range(total_pages),
            format_func=lambda x: f"Page {x + 1}",
        )
        st.session_state[f"{mode}_current_page"] = selected_page

        paginated_image_keys = paginate_images(image_keys, page_size, current_page)
        cols = st.columns(3)

        for idx, img_key in enumerate(paginated_image_keys):
            img_bytes = load_image_from_s3(s3, img_key)
            image = Image.open(BytesIO(img_bytes))
            with cols[idx % 3]:
                st.image(
                    image, caption=os.path.basename(img_key), use_column_width=True
                )
