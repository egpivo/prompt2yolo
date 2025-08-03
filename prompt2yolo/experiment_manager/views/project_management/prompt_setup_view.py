from pathlib import Path

import streamlit as st
import yaml
from filelock import FileLock

PROMPT_FILE_PATH = Path("configs/input.yaml")
LOCK_FILE_PATH = PROMPT_FILE_PATH.with_suffix(".lock")


def load_prompts():
    """Load prompts from the YAML file into session state with file locking."""
    if PROMPT_FILE_PATH.exists():
        with FileLock(LOCK_FILE_PATH):
            with PROMPT_FILE_PATH.open("r") as prompt_file:
                st.session_state["prompt_data"] = yaml.safe_load(prompt_file)
    else:
        st.session_state["prompt_data"] = {"classes": ["person"], "prompts": []}


def save_prompts():
    """Save prompts from session state to the YAML file with file locking."""
    with FileLock(LOCK_FILE_PATH):
        with PROMPT_FILE_PATH.open("w") as prompt_file:
            yaml.dump(
                st.session_state["prompt_data"], prompt_file, default_flow_style=False
            )


def render():
    """Render the prompt setup interface."""
    st.header("Prompt Setup")

    # Load prompts into session state if not already loaded
    if "prompt_data" not in st.session_state:
        load_prompts()

    prompts = st.session_state["prompt_data"].get("prompts", [])

    # Display a table-like interface for existing prompts
    st.subheader("Current Prompts")
    for idx, prompt in enumerate(prompts):
        cols = st.columns([3, 1, 1])  # Three columns: Prompt, Weight, Delete
        with cols[0]:
            prompt["text"] = st.text_input(
                f"Prompt {idx + 1}",
                value=prompt["text"],
                key=f"prompt_text_{idx}",
            )
        with cols[1]:
            prompt["weight"] = st.number_input(
                f"Weight {idx + 1}",
                min_value=0.0,
                max_value=10.0,
                value=prompt["weight"],
                key=f"prompt_weight_{idx}",
            )
        with cols[2]:
            if st.button("🗑️", key=f"delete_prompt_{idx}"):
                prompts.pop(idx)
                st.rerun()

    # Add new prompt dynamically
    st.subheader("Add New Prompt")
    if "new_prompts" not in st.session_state:
        st.session_state["new_prompts"] = []

    for idx, prompt in enumerate(st.session_state["new_prompts"]):
        cols = st.columns([3, 1])  # Two columns: Prompt, Weight
        with cols[0]:
            prompt["text"] = st.text_input(
                f"New Prompt {idx + 1}",
                value=prompt["text"],
                key=f"new_prompt_text_{idx}",
            )
        with cols[1]:
            prompt["weight"] = st.number_input(
                f"New Weight {idx + 1}",
                min_value=0.0,
                max_value=10.0,
                value=prompt["weight"],
                key=f"new_prompt_weight_{idx}",
            )

    # Add button to dynamically create new input fields
    if st.button("➕ Add New Prompt"):
        st.session_state["new_prompts"].append({"text": "", "weight": 1.0})
        st.rerun()

    if st.button("Save Prompts"):
        prompts.extend(st.session_state["new_prompts"])
        st.session_state["prompt_data"]["prompts"] = prompts
        st.session_state["new_prompts"] = []
        save_prompts()
        st.success("Prompts saved successfully!")
