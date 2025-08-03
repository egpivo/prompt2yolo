#!/bin/bash
#
# YOLO Pipeline - Label Evaluation Script
#
# Automates the label evaluation process of the YOLO pipeline using predefined YAML
# configuration files. Supports optional Conda environment activation skipping.
#
# Arguments:
#   --model        Specify the model type ('yolo_v5' or 'yolo_v3_tiny'). Default is 'yolo_v5'.
#   --skip_conda   Skip Conda environment activation (optional).
#   --help, -h     Display this help message.
#
# Usage Example:
#   ./run_evaluate_labels.sh --model yolo_v5 --skip_conda
#

# Load environment variables and utility functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/.bin/init.sh"
source "${SCRIPT_DIR}/.bin/utils.sh"
PACKAGE_DIR="${SCRIPT_DIR}/.."
# Function to display help message
show_help() {
    cat << EOF
Usage: $0 [options]

Options:
  --model        Specify the model type ('yolo_v5' or 'yolo_v3_tiny'). Default is 'yolo_v5'.
  --skip_conda   Skip Conda environment activation (optional).
  --help, -h     Display this help message.
EOF
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
	--model_iteration)
            MODEL_ITERATION="$2"
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

# Determine the configuration file path
INPUT_YAML="${PACKAGE_DIR}/configs/input.yaml"


# Determine the Conda environment and YAML path based on the model type
case "${MODEL}" in
    yolo_v5)
        CONDA_ENV_NAME="${CONDA_YOLO_V5}"
        ;;
    yolo_v3_tiny)
        CONDA_ENV_NAME="${CONDA_YOLO_V3_TINY}"
        ;;
    *)
        echo -e "\e[31m[!] Error: Unknown model type '${MODEL}'. Expected 'yolo_v5' or 'yolo_v3_tiny'.\e[0m"
        exit 1
        ;;
esac

# Activate Conda environment unless skipped
if [[ "${SKIP_CONDA}" == "TRUE" ]]; then
    echo -e "\e[33m[!] Skipping Conda environment activation as per --skip_conda flag.\e[0m"
    pip install .
else
    update_and_activate_conda_env "${CONDA_ENV_NAME}"
fi

# Fetch the first model iteration if not specified
if [[ -z "$MODEL_ITERATION" ]]; then
    echo "[*] Model iteration not specified. Fetching the first iteration..."
    MODEL_ITERATION=1
fi


# Display configuration details
echo -e "\e[90m[*] Running label evaluation with the following configuration:\e[0m"
echo -e "\e[90m    Model: ${MODEL}\e[0m"

# Execute the label evaluation process
bash "${SCRIPT_DIR}/components/evaluate_labels.sh" "${INPUT_YAML}" "${MODEL_ITERATION}"

# Check for successful execution
if [[ $? -ne 0 ]]; then
    echo -e "\e[31m[!] Label evaluation failed.\e[0m"
    exit 1
else
    echo -e "\e[32m[*] Label evaluation completed successfully.\e[0m"
fi

# Completion message
echo -e "\e[32m[*] Script execution completed.\e[0m"
