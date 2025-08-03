import streamlit as st

from prompt2yolo.experiment_manager.utils import get_selected_iteration
from prompt2yolo.experiment_manager.views.evaluation import (
    label_detection_view,
    model_performance_view,
)


def handle_evaluation_action(s3):
    selected_iteration = get_selected_iteration()
    evaluation_tab = st.sidebar.radio(
        "Evaluation Insights",
        ["Model Performance", "Label Detection Analysis"],
    )

    if evaluation_tab == "Model Performance":
        st.title("Model Performance")
        model_performance_view.render(
            s3,
            st.session_state["project_name"],
            selected_iteration,
        )
    elif evaluation_tab == "Label Detection Analysis":
        st.title("Label Detection Analysis")
        label_detection_view.render(
            s3,
            st.session_state["project_name"],
            selected_iteration,
        )
