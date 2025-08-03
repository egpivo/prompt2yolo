#!/bin/bash
#
# YOLOv3-Tiny Training Script
# Trains the YOLOv3-tiny model and uploads the trained model to an S3 bucket.
#
# Arguments:
#   $1 - S3_MODEL_NAME (required): Name for organizing the model on S3.
#
# Usage:
#   ./train_yolov3_tiny.sh <path_to_obj_data> <s3_model_name>
#

set -e  # Exit on any error

CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="${CURRENT_DIR}/../../.."
source "${PACKAGE_DIR}/scripts/.bin/init.sh"
source "${PACKAGE_DIR}/scripts/.bin/utils.sh"

OBJ_DATA_FILENAME="data/obj.data"
S3_MODEL_NAME="$1"

# Check required arguments
if [ -z "$1" ]; then
    echo -e "${FG_RED}[!] Missing required arguments: S3_MODEL_NAME.${FG_RESET}"
    echo -e "Usage: $0 <S3_MODEL_NAME>"
    exit 1
fi

# Define paths
TRAINING_CONFIG="${PACKAGE_DIR}/darknet/cfg/yolov3-tiny_training.cfg"
DARKNET_EXEC="${PACKAGE_DIR}/darknet/darknet"
CONV_FILE="${PACKAGE_DIR}/darknet/yolov3-tiny.conv.15"

# Train YOLOv3-Tiny model
train_yolo() {
    echo -e "${FG_BLUE}[*] Starting YOLOv3-tiny training...${FG_RESET}"
    pushd "${PACKAGE_DIR}/darknet" > /dev/null
    "$DARKNET_EXEC" detector train "${OBJ_DATA_FILENAME}" "${TRAINING_CONFIG}" "${CONV_FILE}" -dont_show

    if [ $? -eq 0 ]; then
        echo -e "${FG_GREEN}[*] Training completed successfully.${FG_RESET}"
    else
        echo -e "${FG_RED}[!] Training failed.${FG_RESET}"
        exit 1
    fi
    popd > /dev/null
}

# Upload trained model to S3
save_model() {
    echo -e "${FG_GRAY}[*] Saving the trained model...${FG_RESET}"
    SAVE_PATH="${PACKAGE_DIR}/darknet/backup/"
    S3_PATH="s3://${AWS_S3_BUCKET_NAME}/projects/${PROJECT}/models/${S3_MODEL_NAME}/"

    echo -e "${FG_BLUE}[*] Uploading model to S3...${FG_RESET}"
    aws s3 cp "${SAVE_PATH}" "${S3_PATH}" --recursive --endpoint-url "${AWS_S3_ENDPOINT}"

    if [ $? -eq 0 ]; then
        echo -e "${FG_GREEN}[*] Model successfully uploaded to ${S3_PATH}${FG_RESET}"
    else
        echo -e "${FG_RED}[!] Failed to upload model to S3.${FG_RESET}"
        exit 1
    fi
}

# Run training and upload functions
train_yolo
save_model
