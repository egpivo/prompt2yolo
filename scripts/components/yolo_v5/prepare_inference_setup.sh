#!/bin/bash
#
# YOLOv5 Inference Preparation Script
# Automates the preparation of data and models for YOLOv5 inference, with options for downloading files from S3.
#
# Arguments:
#   $1 - INPUT_YAML (required): Path to the input YAML configuration file.
#   $2 - DOES_DOWNLOAD_DATA_FROM_S3 (optional): Whether to download data from S3 ("TRUE" or "FALSE"). Defaults to "FALSE".
#   $3 - DOES_DOWNLOAD_MODEL_FROM_S3 (optional): Whether to download model files from S3 ("TRUE" or "FALSE"). Defaults to "FALSE".
#   $4 - ITERATION (optional): Iteration number for managing paths. Defaults to 1.
#
# Usage Example:
#   bash prepare_inference_setup.sh configs/input.yaml TRUE TRUE 1
#
# Features:
#   - Downloads data and models from S3 if specified.
#   - Prepares directories for YOLOv5 inference with test images and labels.
#   - Validates input arguments and ensures all required files are present.
#

CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="${CURRENT_DIR}/../../.."
source "${PACKAGE_DIR}/scripts/.bin/init.sh"
source "${PACKAGE_DIR}/scripts/.bin/utils.sh"


INPUT_YAML="$1"
DOES_DOWNLOAD_DATA_FROM_S3="${2:-FALSE}"
DOES_DOWNLOAD_MODEL_FROM_S3="${3:-FALSE}"
ITERATION="${4:-1}"

if [ -z "${INPUT_YAML}" ]; then
    echo -e "${FG_RED}[!] Missing required argument: INPUT_YAML.${FG_RESET}"
    exit 1
fi

TESTING_DATA_CONFIG="test_data.yaml"
YOLO_DATA_PATH="${PACKAGE_DIR}/yolov5/local_temp/data/yolo/data/iteration_${ITERATION}"
YOLO_MODEL_PATH="${PACKAGE_DIR}/yolov5/runs/train/model/weights/"


prepare_data() {
    echo -e "${FG_BLUE}[*] Preparing YOLOv5 inference data...${FG_RESET}"
    PYTHON_CMD="python ${PACKAGE_DIR}/prompt2yolo/execution/run_yolo_v5_inference_preparation.py \
        --input_yaml \"${INPUT_YAML}\" \
        --testing_data_yaml_filename \"${TESTING_DATA_CONFIG}\" \
        --iteration \"${ITERATION}\""

    [ "${DOES_DOWNLOAD_DATA_FROM_S3}" == "TRUE" ] && PYTHON_CMD+=" --does_download_data_from_s3"
    [ "${DOES_DOWNLOAD_MODEL_FROM_S3}" == "TRUE" ] && PYTHON_CMD+=" --does_download_model_from_s3"

    eval "${PYTHON_CMD}" && \
        echo -e "${FG_GREEN}[*] Data preparation completed successfully.${FG_RESET}" || \
        { echo -e "${FG_RED}[!] Data preparation failed.${FG_RESET}"; exit 1; }
}

prepare_directories() {
    echo -e "${FG_YELLOW}[*] Setting up directories for test set...${FG_RESET}"

    SRC_IMAGE_PATH="${LOCAL_DATA_PATH}/yolo/data/iteration_${ITERATION}/test/images"
    SRC_LABEL_PATH="${LOCAL_DATA_PATH}/yolo/data/iteration_${ITERATION}/test/labels"
    SRC_MODEL_PATH="${LOCAL_DATA_PATH}/yolo/models/iteration_${ITERATION}"
    TARGET_IMAGE_PATH="${YOLO_DATA_PATH}/test/images"
    TARGET_LABEL_PATH="${YOLO_DATA_PATH}/test/labels"

    clean_directory "${TARGET_IMAGE_PATH}"
    clean_directory "${TARGET_LABEL_PATH}"
    clean_directory "${YOLO_MODEL_PATH}"

    mkdir -p "${TARGET_IMAGE_PATH}" "${TARGET_LABEL_PATH}"

    for SRC_PATH in "${SRC_IMAGE_PATH}" "${SRC_MODEL_PATH}"; do
        if [ ! -d "${SRC_PATH}" ] || [ -z "$(find "${SRC_PATH}" -type f)" ]; then
            echo -e "${FG_RED}[!] Missing or empty source directory: ${SRC_PATH}.${FG_RESET}"
            exit 1
        fi
    done
    cp -r "${SRC_IMAGE_PATH}/"* "${TARGET_IMAGE_PATH}/"
    cp -r "${SRC_MODEL_PATH}/"* "${YOLO_MODEL_PATH}"

    # Check if labels are missing; if so, treat all images as negative class
    if [ -z "$(find "${SRC_LABEL_PATH}" -type f)" ]; then
        echo -e "${FG_YELLOW}[!] Label directory is empty. Treating all images as negative class.${FG_RESET}"
    else
         cp -r "${SRC_LABEL_PATH}/"* "${TARGET_LABEL_PATH}/"
    fi

    echo -e "${FG_GREEN}[*] Test data and model setup completed in ${YOLO_DATA_PATH}.${FG_RESET}"
}

prepare_data
prepare_directories
