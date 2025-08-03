#!/bin/bash
#
# YOLOv5 Model Data Preparation and Training Pipeline
#
# This script focuses on preparing data and training a YOLOv5 model.
# It loads environment variables, prepares the data, and executes the training script.
#
# Parameters:
#   --does_download_data_from_s3    Optional flag. Include this flag to download data from S3 during preparation.
#   --skip_conda                    Optional flag. Skip Conda environment activation.
#
# Usage Example:
#   ./run_yolo_v5_model_training.sh --does_download_data_from_s3 --skip_conda
#

# Load required environment variables and utility functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/.bin/init.sh"
source "${SCRIPT_DIR}/.bin/utils.sh"
PACKAGE_DIR="${SCRIPT_DIR}/.."

# Function to display help message
show_help() {
    echo -e "Usage: $0 [--does_download_data_from_s3] [--skip_conda]"
    echo -e "    --does_download_data_from_s3    Optional. Include this flag to download data from S3 during preparation."
    echo -e "    --skip_conda                    Optional. Skip Conda environment activation."
}

# Check for help flags
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    show_help
    exit 0
fi

# Set script paths and configurations
CONFIGS_DIR="${PACKAGE_DIR}/configs"
INPUT_YAML="${CONFIGS_DIR}/input.yaml"
DOES_DOWNLOAD_DATA_FROM_S3="FALSE"
SKIP_CONDA="FALSE"  # Default to not skipping Conda activation

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --does_download_data_from_s3)
            DOES_DOWNLOAD_DATA_FROM_S3="TRUE"
            shift
            ;;
        --skip_conda)
            SKIP_CONDA="TRUE"
            shift
            ;;
        *)
            echo -e "${FG_RED}[!] Error: Unknown parameter: $1${FG_RESET}"
            show_help
            exit 1
            ;;
    esac
done

# Conditionally activate Conda environment unless --skip_conda is provided
if [[ "${SKIP_CONDA}" == "TRUE" ]]; then
    echo -e "${FG_YELLOW}[!] Skipping Conda environment activation as per --skip_conda flag.${FG_RESET}"
else
    update_and_activate_conda_env "${CONDA_YOLO_V5}"
fi

MODEL_VERSION=$(get_new_model_version)
S3_MODEL_NAME="model_${MODEL_VERSION}"

# Display parameter values
echo -e "${FG_GRAY}[*] Running YOLOv5 model data preparation and training with the following parameters:${FG_RESET}"
echo -e "${FG_GRAY}    Input YAML: ${INPUT_YAML}${FG_RESET}"
echo -e "${FG_GRAY}    Training Config YAML: ${TRAINING_CONFIG_YAML}${FG_RESET}"
echo -e "${FG_GRAY}    Download from S3: ${DOES_DOWNLOAD_DATA_FROM_S3}${FG_RESET}"
echo -e "${FG_GRAY}    Skip Conda Activation: ${SKIP_CONDA}${FG_RESET}"

# Start the data preparation
echo -e "${FG_YELLOW}[*] Starting YOLOv5 data preparation...${FG_RESET}"

# Install dependencies (if needed)
bash "${SCRIPT_DIR}/components/yolo_v5/install_dependencies.sh"

# Prepare data for YOLOv5 training
bash "${SCRIPT_DIR}/components/yolo_v5/prepare_training_setup.sh" "${INPUT_YAML}" "${DOES_DOWNLOAD_DATA_FROM_S3}"

# Start model training
echo -e "${FG_BLUE}[*] Starting YOLOv5 model training...${FG_RESET}"
bash "${SCRIPT_DIR}/components/yolo_v5/train_model.sh" "${S3_MODEL_NAME}"

# Completion message
echo -e "${FG_GREEN}[*] YOLOv5 data preparation and model training completed successfully.${FG_RESET}"
