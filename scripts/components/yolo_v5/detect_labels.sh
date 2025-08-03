#!/bin/bash
#
# YOLOv5 Detection Script
# Automates YOLOv5 detection on input data and evaluates label predictions, uploading results to S3.
#
# Arguments:
#   $1 - ITERATION (optional): Iteration number for managing paths. Defaults to 1.
#
# Usage Example:
#   bash detect_label.sh 1
#
# Prerequisites:
# - AWS CLI configured with valid credentials.
# - Detection configuration file available at specified location.

# Load environment variables and utility functions
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="${CURRENT_DIR}/../../.."
source "${PACKAGE_DIR}/scripts/.bin/init.sh"
source "${PACKAGE_DIR}/scripts/.bin/utils.sh"

# Arguments
ITERATION="${1:-1}"
DETECTION_CONFIG_FILE="${PACKAGE_DIR}/configs/yolo_v5/evaluation.yaml"
YOLO_DATA_PATH="${PACKAGE_DIR}/yolov5/local_temp/data/yolo/data/iteration_${ITERATION}"
DETECTION_MODEL="${PACKAGE_DIR}/yolov5/runs/train/model/iteration_${ITERATION}/weights/best.pt"


# Extract detection configuration values
TESTING_DATA_PATH="${YOLO_DATA_PATH}/$(get_yaml_value "data_path" "${DETECTION_CONFIG_FILE}")"
CONFIDENCE=$(get_yaml_value "confidence" "${DETECTION_CONFIG_FILE}")
IOU=$(get_yaml_value "iou" "${DETECTION_CONFIG_FILE}")
IMG_SIZE=$(get_yaml_value "img_size" "${DETECTION_CONFIG_FILE}")
DETECTION_DIR_NAME=$(get_yaml_value "directory" "${DETECTION_CONFIG_FILE}")

# Validate critical environment variables
for var in PACKAGE_DIR LOCAL_DATA_PATH AWS_S3_BUCKET_NAME PROJECT; do
  if [ -z "${!var}" ]; then
    echo -e "\e[31m[!] Error: ${var} environment variable is not set.\e[0m"
    exit 1
  fi
done

run_detect() {
    echo -e "\e[34m[*] Starting YOLOv5 detection...\e[0m"

    # Prepare the output directory
    DETECTION_OUTPUT_DIR="${PACKAGE_DIR}/yolov5/runs/detect"
    rm -rf "${DETECTION_OUTPUT_DIR}" && mkdir -p "${DETECTION_OUTPUT_DIR}"

    # Run YOLOv5 detection
    python "${PACKAGE_DIR}/yolov5/detect.py" \
      --img "${IMG_SIZE}" \
      --conf "${CONFIDENCE}" \
      --iou "${IOU}" \
      --weights "${DETECTION_MODEL}" \
      --source "${TESTING_DATA_PATH}" \
      --project "${DETECTION_OUTPUT_DIR}" \
      --save-txt \
      --name "${DETECTION_DIR_NAME}" 2>&1 | tee "detection_${ITERATION}.log"

    if [ $? -eq 0 ]; then
        echo -e "\e[32m[*] Detection completed successfully.\e[0m"
        mkdir -p "${LOCAL_DATA_PATH}/yolo/data/iteration_${ITERATION}/test/predicted_labels"

        # Check if there are any files to copy
        if [ "$(find "${DETECTION_OUTPUT_DIR}/${DETECTION_DIR_NAME}/labels/" -type f | wc -l)" -gt 0 ]; then
            cp -r "${DETECTION_OUTPUT_DIR}/${DETECTION_DIR_NAME}/labels/"* "${LOCAL_DATA_PATH}/yolo/data/iteration_${ITERATION}/test/predicted_labels/"
            echo -e "\e[32m[*] Predicted labels copied successfully.\e[0m"
        else
            echo -e "\e[33m[!] No predicted labels to copy. Skipping.\e[0m"
        fi
    else
        echo -e "\e[31m[!] Detection failed. Check detection_${ITERATION}.log for details.\e[0m"
        exit 1
    fi
}

save_results() {
    echo -e "\e[34m[*] Saving detection results...\e[0m"

    SAVE_PATH="${PACKAGE_DIR}/yolov5/runs/detect/${DETECTION_DIR_NAME}/"
    S3_PATH="s3://${AWS_S3_BUCKET_NAME}/projects/${PROJECT}/models/iteration_${ITERATION}/evaluation/label/detection"

    if ! command -v aws &>/dev/null; then
        echo -e "\e[31m[!] Error: aws-cli is not installed.\e[0m"
        exit 1
    fi
    INCLUDES=("*.png" "*.jpg")  # Define inclusion patterns

    upload_to_s3 "${SAVE_PATH}" "${S3_PATH}" "${INCLUDES[@]}" 2>&1 | tee "s3_upload_${ITERATION}.log"

    if [ $? -eq 0 ]; then
        echo -e "\e[32m[*] Results uploaded to ${S3_PATH}.\e[0m"
    else
        echo -e "\e[31m[!] Failed to upload results. Check s3_upload_${ITERATION}.log for details.\e[0m"
        exit 1
    fi
}

# Execute detection and upload results
run_detect && save_results
