#!/bin/bash

# Common initialization script
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="${DIR}/../.."

# Export all variables from the .env file
set -a
source "${PACKAGE_DIR}/.env"
set +a

# Load color map settings
source "${DIR}/color_map.sh"

# Internal use file names
TRAINING_DATA_CONFIG="train_data.yaml"
TESTING_DATA_CONFIG="test_data.yaml"
HYP_CONFIG="hyp.yaml"

# Internal folder names
TRAINING_DIR_NAME="model"
TESTING_DIR_NAME="model"

# TRAINING CONFIG
OPTIMIZER="Adam"
PATIENCE=100

# INFERENCE CONFIG
CONFIDENCE=0.05
IOU=0.4
IMG_SIZE=512

CONDA_YOLO_V5="prompt2yolo_v5"
CONDA_YOLO_V3_TINY="prompt2yolo_v3_tiny"


# Debugging: Print the environment variables to verify they’re loaded
echo "LOCAL_DATA_PATH is set to: ${LOCAL_DATA_PATH}"
echo "AWS_S3_ENDPOINT is set to: ${AWS_S3_ENDPOINT}"
echo "PROJECT is set to: ${PROJECT}"

# Clean LOCAL_DATA_PATH if it exists
if [ -d "${LOCAL_DATA_PATH}" ]; then
    echo -e "${FG_YELLOW}[!] Removing existing LOCAL_DATA_PATH: ${LOCAL_DATA_PATH}${FG_RESET}"
    rm -rf "${LOCAL_DATA_PATH}" || {
        echo -e "${FG_RED}[!] Failed to remove LOCAL_DATA_PATH.${FG_RESET}"
        exit 1
    }
else
    echo -e "${FG_GREEN}[✓] LOCAL_DATA_PATH does not exist. No need to clean.${FG_RESET}"
fi

# Recreate LOCAL_DATA_PATH directory
mkdir -p "${LOCAL_DATA_PATH}" || {
    echo -e "${FG_RED}[!] Failed to create LOCAL_DATA_PATH directory.${FG_RESET}"
    exit 1
}
echo -e "${FG_GREEN}[✓] LOCAL_DATA_PATH cleaned and ready: ${LOCAL_DATA_PATH}${FG_RESET}"
