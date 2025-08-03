import streamlit as st

from prompt2yolo.experiment_manager.utils import get_selected_iteration
from prompt2yolo.experiment_manager.views.dataset import dataset_view


def handle_dataset_action(s3):
    selected_iteration = get_selected_iteration()
    st.title("Dataset Overview")
    dataset_view.render(
        s3,
        st.session_state["project_name"],
        selected_iteration,
    )
