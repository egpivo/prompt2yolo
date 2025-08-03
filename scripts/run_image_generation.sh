#!/bin/bash
#
# YOLO Pipeline - Image Generation Script with Validation and Test Ratios
#
# Automates the image generation part of the YOLO pipeline using predefined YAML
# configuration files. Supports optional validation/test ratios and Conda activation.
#
# Arguments:
#   --model        Specify the model (default: yolo_v5).
#   --skip_conda   Skip Conda environment activation (optional).
#   --help, -h     Display this help message.
#
# Usage Example:
#   ./run_image_generation.sh --model yolo_v5 --skip_conda
#

# Load environment variables and utility functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/.bin/init.sh"
source "${SCRIPT_DIR}/.bin/utils.sh"

# Function to display help message
show_help() {
    echo -e "Usage: $0 [options]"
    echo -e "Options:"
    echo -e "    --model        Specify the model (default: yolo_v5)."
    echo -e "    --skip_conda   Skip Conda environment activation (optional)."
    echo -e "    --help, -h     Display this help message."
}

# Default argument values
MODEL="yolo_v5"
SKIP_CONDA="FALSE"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --skip_conda)
            SKIP_CONDA="TRUE"
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            echo -e "\e[31m[!] Error: Unknown parameter: $1\e[0m"
            show_help
            exit 1
            ;;
    esac
done

if [[ "${MODEL}" == "yolo_v5" ]]; then
    CONDA_ENV_NAME="${CONDA_YOLO_V5}"
    TESTING_CONFIG_YAML="${CONFIGS_DIR}/${MODEL}/testing.yaml"
elif [[ "${MODEL}" == "yolo_v3_tiny" ]]; then
    CONDA_ENV_NAME="${CONDA_YOLO_V3_TINY}"
    TESTING_CONFIG_YAML="${CONFIGS_DIR}/${MODEL}/testing.yaml"
else
    echo "Error: Unknown MODEL value '${MODEL}'. Expected 'yolo_v5' or 'yolo_v3_tiny'."
    exit 1
fi

if [[ "${SKIP_CONDA}" == "TRUE" ]]; then
    echo -e "\e[33m[!] Skipping Conda environment activation as per --skip_conda flag.\e[0m"
    pip install .
else
    update_and_activate_conda_env "${CONDA_ENV_NAME}"
fi

# Set configuration paths
CONFIGS_DIR="${SCRIPT_DIR}/../configs"
INPUT_YAML="${CONFIGS_DIR}/input.yaml"
IMAGE_GENERATOR_YAML="${CONFIGS_DIR}/${MODEL}/image_generator.yaml"
IMAGE_LABELER_YAML="${CONFIGS_DIR}/${MODEL}/image_labeler.yaml"

# Display configuration details
echo -e "\e[90m[*] Running image generation with the following configurations:\e[0m"
echo -e "\e[90m    Input YAML: ${INPUT_YAML}\e[0m"
echo -e "\e[90m    Image Generator YAML: ${IMAGE_GENERATOR_YAML}\e[0m"
echo -e "\e[90m    Image Labeler YAML: ${IMAGE_LABELER_YAML}\e[0m"

# Execute the image generation process
bash "${SCRIPT_DIR}/components/generate_image_data.sh" \
    "${INPUT_YAML}" "${IMAGE_GENERATOR_YAML}" "${IMAGE_LABELER_YAML}"

# Check for successful execution
if [[ $? -ne 0 ]]; then
    echo -e "\e[31m[!] Image generation failed.\e[0m"
    exit 1
else
    echo -e "\e[32m[*] Image generation completed successfully.\e[0m"
fi

# Completion message
echo -e "\e[32m[*] Script execution completed.\e[0m"
