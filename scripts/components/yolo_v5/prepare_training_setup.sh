#!/bin/bash

# YOLOv5 Training Preparation Script
# This script prepares data for YOLO training by running a Python data preparation script and optionally downloading data from S3.
#
# Arguments:
#   $1 - INPUT_YAML (required): Path to the input YAML file containing class and configuration details.
#   $2 - DOES_DOWNLOAD_DATA_FROM_S3 (optional): Flag to trigger data download from S3 ("TRUE" or "FALSE"). Defaults to "FALSE".
#   $3 - ITERATION (optional): Iteration number for training. Defaults to "1".
#
# Usage Example:
#   bash scripts/components/yolo_v5/prepare_training_setup.sh configs/input.yaml TRUE 1
#

# Define the directory of the script and source environment variables
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="${CURRENT_DIR}/../../.."
source "${PACKAGE_DIR}/scripts/.bin/init.sh"

# Parse and assign arguments
INPUT_YAML="${1}"
DOES_DOWNLOAD_DATA_FROM_S3="${2:-FALSE}"
ITERATION="${3:-1}"

# Check if the required argument INPUT_YAML is provided
if [ -z "${INPUT_YAML}" ]; then
    echo -e "${FG_RED}[!] Error: Missing required argument INPUT_YAML.${FG_RESET}"
    echo -e "Usage: $0 <input_yaml> [TRUE|FALSE] [iteration]"
    exit 1
fi

# Internal filenames for configuration
TRAINING_DATA_CONFIG="train_data.yaml"
HYP_CONFIG="hyp.yaml"
YOLO_DATA_PATH="${PACKAGE_DIR}/yolov5/local_temp/data/yolo/data/iteration_${ITERATION}"

prepare_data() {
    echo -e "${FG_BLUE}[*] Preparing YOLO training data...${FG_RESET}"

    # Construct the Python command for data preparation
    local PYTHON_CMD="python ${PACKAGE_DIR}/prompt2yolo/execution/run_yolo_v5_training_preparation.py \
        --input_yaml \"${INPUT_YAML}\" \
        --training_data_yaml_filename \"${TRAINING_DATA_CONFIG}\" \
        --hyp_config_name \"${HYP_CONFIG}\" \
        --iteration \"${ITERATION}\""

    # Append the S3 download flag if specified
    if [ "${DOES_DOWNLOAD_DATA_FROM_S3}" == "TRUE" ]; then
        PYTHON_CMD="${PYTHON_CMD} --does_download_data_from_s3"
        echo -e "${FG_BLUE}[*] Downloading data from S3...${FG_RESET}"
    fi

    # Execute the Python command
    eval "${PYTHON_CMD}"

    # Confirm data preparation completion
    if [ $? -eq 0 ]; then
        echo -e "${FG_GREEN}[*] YOLO data preparation completed successfully.${FG_RESET}"
    else
        echo -e "${FG_RED}[!] YOLO data preparation failed.${FG_RESET}"
        exit 1
    fi
}

prepare_directories() {
    echo -e "${FG_YELLOW}[*] Preparing YOLO directories for iteration ${ITERATION}...${FG_RESET}"

    SRC_PATH="${LOCAL_DATA_PATH}/yolo/data/iteration_${ITERATION}"
    TARGET_PATH="${YOLO_DATA_PATH}"

    for MODE in "train" "val"; do
        mkdir -p "${TARGET_PATH}/${MODE}/images" "${TARGET_PATH}/${MODE}/labels"

        # Copy files from source to target
        cp -r "${SRC_PATH}/${MODE}/images/"* "${TARGET_PATH}/${MODE}/images/"
        cp -r "${SRC_PATH}/${MODE}/labels/"* "${TARGET_PATH}/${MODE}/labels/"
    done

    echo -e "${FG_GREEN}[*] YOLO directories prepared successfully at ${YOLO_DATA_PATH}.${FG_RESET}"
}

# Execute the data preparation and directory setup
prepare_data
prepare_directories
