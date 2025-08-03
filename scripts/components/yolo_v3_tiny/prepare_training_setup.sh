#!/bin/bash

# YOLOv3-Tiny Training Preparation Script
# Prepares data for YOLOv3-tiny training by running a Python script, handling data download from S3,
# and organizing image and label files for training.

# Arguments:
#   $1 - INPUT_YAML (required): Path to the YAML file containing class and config details.
#   $2 - DOES_DOWNLOAD_DATA_FROM_S3 (optional): Flag to trigger data download from S3 ("TRUE" or "FALSE"). Default: FALSE.
#   $3 - ITERATION (optional): Iteration index for the training data. Default: 1.

CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="${CURRENT_DIR}/../../.."
source "${PACKAGE_DIR}/scripts/.bin/init.sh"

# Arguments
INPUT_YAML="${1}"
DOES_DOWNLOAD_DATA_FROM_S3="${2:-FALSE}"  # Default to FALSE
ITERATION="${3:-1}"

# Path configurations
IMAGE_FOLDER_PATH="${PACKAGE_DIR}/${LOCAL_DATA_PATH}/yolo/data/iteration_${ITERATION}/train/images"
LABEL_FOLDER_PATH="${PACKAGE_DIR}/${LOCAL_DATA_PATH}/yolo/data/iteration_${ITERATION}/train/labels"
DEST_IMAGE_FOLDER_PATH="${PACKAGE_DIR}/darknet/data/iteration_${ITERATION}/images"

# Internal file paths
SAVE_MODEL_PATH="${PACKAGE_DIR}/darknet/model.weights"
CFG_TEMPLATE_PATH="${PACKAGE_DIR}/darknet/cfg/yolov3-tiny.cfg"
OUTPUT_CFG_PATH="${PACKAGE_DIR}/darknet/cfg/yolov3-tiny_training.cfg"
OBJ_NAMES_FILENAME="${PACKAGE_DIR}/darknet/data/obj.names"
OBJ_DATA_FILENAME="${PACKAGE_DIR}/darknet/data/obj.data"
TRAIN_TXT_FILENAME="${PACKAGE_DIR}/darknet/data/train.txt"

# Check for required arguments
if [ -z "${INPUT_YAML}" ]; then
    echo -e "[Error] Missing argument: INPUT_YAML. Usage: ./prepare_yolov3_tiny_setup.sh <input_yaml> [TRUE|FALSE] [iteration]"
    exit 1
fi

prepare_data() {
    echo "[*] Starting YOLOv3-tiny data preparation..."
    mkdir -p "${SAVE_MODEL_PATH}"

    # Prepare Python command
    PYTHON_CMD="python ${PACKAGE_DIR}/prompt2yolo/execution/run_yolo_v3_tiny_training_preparation.py \
        --input_yaml \"${INPUT_YAML}\" \
        --obj_names_filename \"${OBJ_NAMES_FILENAME}\" \
        --obj_data_filename \"${OBJ_DATA_FILENAME}\" \
        --train_txt_filename \"${TRAIN_TXT_FILENAME}\" \
        --save_model_path \"${SAVE_MODEL_PATH}\" \
        --cfg_template_path \"${CFG_TEMPLATE_PATH}\" \
        --output_cfg_path \"${OUTPUT_CFG_PATH}\""

    # Append S3 download flag if requested
    if [ "${DOES_DOWNLOAD_DATA_FROM_S3}" == "TRUE" ]; then
        PYTHON_CMD+=" --does_download_data_from_s3"
        echo "[*] Downloading data from S3..."
    fi

    # Execute Python command
    eval "${PYTHON_CMD}"

    # Verify directories
    if [ ! -d "${IMAGE_FOLDER_PATH}" ] || [ ! -d "${LABEL_FOLDER_PATH}" ]; then
        echo "[Error] Missing required folders: ${IMAGE_FOLDER_PATH} or ${LABEL_FOLDER_PATH}."
        exit 1
    fi

    # Organize data
    echo "[*] Copying images and labels to training directory..."
    mkdir -p "${DEST_IMAGE_FOLDER_PATH}"

    # Copy image files
    IMG_EXTENSIONS=("jpg" "jpeg" "png")
    for ext in "${IMG_EXTENSIONS[@]}"; do
        cp "${IMAGE_FOLDER_PATH}"/*.${ext} "${DEST_IMAGE_FOLDER_PATH}" 2>/dev/null || true
    done

    # Copy label files
    cp "${LABEL_FOLDER_PATH}"/* "${DEST_IMAGE_FOLDER_PATH}" 2>/dev/null || true

    # Generate train.txt
    echo "[*] Writing image paths to train.txt..."
    rm -f "${TRAIN_TXT_FILENAME}"
    for ext in "${IMG_EXTENSIONS[@]}"; do
        for IMG_FILE in "${DEST_IMAGE_FOLDER_PATH}"/*.${ext}; do
            [ -f "${IMG_FILE}" ] && echo "${IMG_FILE}" >> "${TRAIN_TXT_FILENAME}"
        done
    done

    echo "[*] train.txt created at: ${TRAIN_TXT_FILENAME}"
    echo "[*] YOLOv3-tiny data preparation completed successfully."
}

prepare_data
