import matplotlib.pyplot as plt
import yaml

from prompt2yolo.utils.logger import setup_logger

LOGGER = setup_logger(__name__)


def plot_prompt_weights(
    weights: dict, truncate_length: int = 50, save_path: str = None
):
    """Plots and optionally saves prompt weights."""
    truncated_prompts = [
        prompt[:truncate_length] + "..." if len(prompt) > truncate_length else prompt
        for prompt in weights.keys()
    ]
    weight_values = list(weights.values())

    plt.figure(figsize=(12, 6))
    plt.barh(truncated_prompts, weight_values, color="skyblue")
    plt.xlabel("Weights (Inverse FP Count)")
    plt.ylabel("Prompts")
    plt.title("Prompt Weights Based on False Positives")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        LOGGER.info(f"Plot saved to {save_path}")

    plt.show()


def plot_fp_distribution(
    fp_counts: dict, truncate_length: int = 50, save_path: str = None
):
    """Plots and optionally saves the false positive distribution."""
    truncated_prompts = [
        prompt[:truncate_length] + "..." if len(prompt) > truncate_length else prompt
        for prompt in fp_counts.keys()
    ]
    fp_values = list(fp_counts.values())

    plt.figure(figsize=(12, 6))
    plt.barh(truncated_prompts, fp_values, color="salmon")
    plt.xlabel("False Positive Rate")
    plt.ylabel("Prompts")
    plt.title("Distribution of False Positive Rates by Prompt")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        LOGGER.info(f"Plot saved to {save_path}")

    plt.show()


def overwrite_input_yaml(input_yaml_path: str, prompt_weights: dict):
    """Overwrite the input YAML file with updated prompt weights."""
    # Load the original input YAML
    with open(input_yaml_path, "r") as file:
        input_config = yaml.safe_load(file)

    # Update weights in prompts
    for prompt in input_config.get("prompts", []):
        if prompt["text"] in prompt_weights:
            prompt["weight"] = prompt_weights[prompt["text"]]

    # Save the updated YAML
    with open(input_yaml_path, "w") as file:
        yaml.safe_dump(input_config, file)

    LOGGER.info(f"Updated {input_yaml_path} with new weights.")
