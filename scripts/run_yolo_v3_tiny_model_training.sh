#!/bin/bash
#
# YOLOv3-Tiny Model Data Preparation and Training Pipeline
#
# This script prepares data and trains a YOLOv3-tiny model by loading environment variables,
# preparing the data, and executing the model training script. The trained model can then be
# saved to an S3 bucket if specified.
#
# Parameters:
#   --does_download_data_from_s3       Optional flag. Use this flag to download necessary training data from S3.
#   --skip_conda                       Optional flag. Skips the activation of the Conda environment if provided.
#
# Usage Example:
#   ./run_yolo_v3_tiny_model_training.sh --does_download_data_from_s3 --skip_conda
#
# Notes:
#   - Ensure that files are stored in the `darknet` folder to avoid path issues.
#   - This script assumes necessary environment variables and functions are defined in
#     sourced files (`init.sh`, `utils.sh`, `color_map.sh`).
#

# Load environment variables and utility functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/.bin/init.sh"
source "${SCRIPT_DIR}/.bin/utils.sh"

# Help message function to explain script usage
show_help() {
    echo -e "Usage: $0 [--does_download_data_from_s3] [--skip_conda]"
    echo -e "    --does_download_data_from_s3       Optional flag to download training data from S3."
    echo -e "    --skip_conda                       Optional flag to skip Conda environment activation."
}

# Display help if --help or -h flag is provided
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    show_help
    exit 0
fi

# Default values for optional parameters
CONFIGS_DIR="${SCRIPT_DIR}/../configs"
INPUT_YAML="${CONFIGS_DIR}/input.yaml"
DOES_DOWNLOAD_DATA_FROM_S3="FALSE"

SKIP_CONDA=false

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --does_download_data_from_s3)
            DOES_DOWNLOAD_DATA_FROM_S3="TRUE"
            shift
            ;;
        --skip_conda)
            SKIP_CONDA=true
            shift
            ;;
        *)
            echo -e "${FG_RED}[!] Error: Unknown parameter: $1${FG_RESET}"
            show_help
            exit 1
            ;;
    esac
done

# Conditionally activate the Conda environment
if [[ "${SKIP_CONDA}" == true ]]; then
    echo -e "${FG_YELLOW}[!] Skipping Conda environment activation as per --skip_conda flag.${FG_RESET}"
else
    update_and_activate_conda_env "${CONDA_YOLO_V3_TINY}"
fi

# Determine model version automatically
MODEL_VERSION="$(get_new_model_version)"
S3_MODEL_NAME="model_${MODEL_VERSION}"

# Display parameter values
echo -e "${FG_GRAY}[*] YOLOv3-Tiny Model Training Pipeline Parameters:${FG_RESET}"
echo -e "${FG_GRAY}    Input YAML: ${INPUT_YAML}${FG_RESET}"
echo -e "${FG_GRAY}    Model Version: ${MODEL_VERSION}${FG_RESET}"
echo -e "${FG_GRAY}    Download from S3: ${DOES_DOWNLOAD_DATA_FROM_S3}${FG_RESET}"
echo -e "${FG_GRAY}    Skip Conda Activation: ${SKIP_CONDA}${FG_RESET}"

# Install dependencies if needed
echo -e "${FG_YELLOW}[*] Installing dependencies...${FG_RESET}"
bash "${SCRIPT_DIR}/components/yolo_v3_tiny/install_dependencies.sh"

# Prepare data for YOLOv3-tiny training
echo -e "${FG_YELLOW}[*] Preparing data for YOLOv3-Tiny training...${FG_RESET}"
mkdir -p "darknet/data"
bash "${SCRIPT_DIR}/components/yolo_v3_tiny/prepare_training_setup.sh" "${INPUT_YAML}" "${DOES_DOWNLOAD_DATA_FROM_S3}"
# Start model training
echo -e "${FG_BLUE}[*] Starting YOLOv3-Tiny model training...${FG_RESET}"
bash "${SCRIPT_DIR}/components/yolo_v3_tiny/train_model.sh" "${S3_MODEL_NAME}"

# Completion message with error check
if [[ $? -eq 0 ]]; then
    echo -e "${FG_GREEN}[*] YOLOv3-Tiny data preparation and model training completed successfully.${FG_RESET}"
else
    echo -e "${FG_RED}[!] YOLOv3-Tiny data preparation and model training failed.${FG_RESET}"
    exit 1
fi
