import streamlit as st
from dotenv import load_dotenv

from prompt2yolo.experiment_manager.controllers.project_controller import (
    handle_project_action,
)
from prompt2yolo.experiment_manager.controllers.training_controller import (
    handle_training_action,
)
from prompt2yolo.experiment_manager.models.project_model import list_projects
from prompt2yolo.experiment_manager.utils import initialize_s3_client

# Load environment variables
load_dotenv()

# Initialize Streamlit app configuration
st.set_page_config(
    page_title="Human Occupancy Detection Manager",
    page_icon="🧑‍💻",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar configuration
st.sidebar.title("Human Occupancy Detection Manager")
st.sidebar.markdown("Choose an action to get started.")

# Initialize S3 client and projects
s3 = initialize_s3_client()
projects = list_projects(s3)

# Initialize session state
if "project_name" not in st.session_state:
    st.session_state["project_name"] = None
if "training_started" not in st.session_state:
    st.session_state["training_started"] = False
if "training_process" not in st.session_state:
    st.session_state["training_process"] = None
if "training_logs" not in st.session_state:
    st.session_state["training_logs"] = []

# Sidebar navigation
action = st.sidebar.radio(
    "Select an Action",
    [
        "Create New Project",
        "Check Existing Project",
        "Ongoing Training Job",
    ],
    key="action",
)

if action == "Create New Project":
    handle_project_action(s3, action, projects)
elif action == "Check Existing Project":
    handle_project_action(s3, action, projects)
elif action == "Ongoing Training Job":
    handle_training_action(action)
