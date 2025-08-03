#!/bin/bash
#
# YOLOv5 Evaluation Script
# Automates the evaluation of YOLOv5 models and uploads results to S3.
#
# Arguments:
#   $1 - ITERATION (optional): Iteration number (default: 1).
#
# Usage:
#   ./evaluate_model.sh 1
#
# Features:
#   - Validates input arguments and configurations.
#   - Extracts test settings from YAML files.
#   - Executes YOLOv5 evaluation and uploads results to S3.

# Load environment and utility scripts
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="${SCRIPT_DIR}/../../.."
source "${PACKAGE_DIR}/scripts/.bin/init.sh"
source "${PACKAGE_DIR}/scripts/.bin/utils.sh"

# Arguments
ITERATION="${1:-1}"

TESTING_CONFIG_FILE="${PACKAGE_DIR}/configs/yolo_v5/evaluation.yaml"
TEST_MODEL="${PACKAGE_DIR}/yolov5/runs/train/model/iteration_${ITERATION}/weights/best.pt"

# Extract configurations
CONFIDENCE=$(get_yaml_value "confidence" "${TESTING_CONFIG_FILE}")
IOU=$(get_yaml_value "iou" "${TESTING_CONFIG_FILE}")
IMG_SIZE=$(get_yaml_value "img_size" "${TESTING_CONFIG_FILE}")
BATCH_SIZE=$(get_yaml_value "batch_size" "${TESTING_CONFIG_FILE}")
TESTING_DIR_NAME=$(get_yaml_value "directory" "${TESTING_CONFIG_FILE}")/iteration_${ITERATION}
TESTING_DATA_CONFIG=$(get_yaml_value "data_config" "${TESTING_CONFIG_FILE}")

# Validate configurations
REQUIRED_VARS=("CONFIDENCE" "IOU" "IMG_SIZE" "BATCH_SIZE" "TESTING_DIR_NAME" "TESTING_DATA_CONFIG")
for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var}" ]; then
        echo -e "\e[31m[!] Error: ${var} is missing in ${TESTING_CONFIG_FILE}.\e[0m"
        exit 1
    fi
done

# Validate environment variables
ENV_VARS=("PACKAGE_DIR" "LOCAL_DATA_PATH" "AWS_S3_BUCKET_NAME" "PROJECT")
for var in "${ENV_VARS[@]}"; do
    if [ -z "${!var}" ]; then
        echo -e "\e[31m[!] Error: ${var} is not set.\e[0m"
        exit 1
    fi
done

DATA_CONFIG_PATH="${LOCAL_DATA_PATH}/yolo/configs/${TESTING_DATA_CONFIG}"

# Ensure critical paths exist
if [ ! -f "${DATA_CONFIG_PATH}" ]; then
    echo -e "\e[31m[!] Error: Data config file ${DATA_CONFIG_PATH} does not exist.\e[0m"
    exit 1
fi

# Run YOLOv5 evaluation
run_evaluate() {
    echo -e "\e[34m[*] Running YOLOv5 evaluation...\e[0m"
    python "${PACKAGE_DIR}/yolov5/val.py" \
        --img "${IMG_SIZE}" \
        --batch "${BATCH_SIZE}" \
        --conf "${CONFIDENCE}" \
        --iou "${IOU}" \
        --weights "${TEST_MODEL}" \
        --data "${DATA_CONFIG_PATH}" \
        --name "${TESTING_DIR_NAME}" 2>&1 | tee evaluation.log

    if [ $? -eq 0 ]; then
        echo -e "\e[32m[*] Evaluation completed successfully.\e[0m"
    else
        echo -e "\e[31m[!] Evaluation failed. Check evaluation.log for details.\e[0m"
        exit 1
    fi
}

# Save and upload evaluation results
save_results() {
    echo -e "\e[34m[*] Uploading results to S3...\e[0m"

    # Define paths
    SAVE_PATH="${PACKAGE_DIR}/yolov5/runs/val/${TESTING_DIR_NAME}/"
    S3_EVAL_PATH="s3://${AWS_S3_BUCKET_NAME}/projects/${PROJECT}/models/iteration_${ITERATION}/evaluation/model/inference/performance"
    S3_INFER_PATH="s3://${AWS_S3_BUCKET_NAME}/projects/${PROJECT}/models/iteration_${ITERATION}/evaluation/model/inference/prediction"

    # Define inclusion lists
    EVAL_INCLUDES=(
        "confusion_matrix.png"
    )
    PREDICTION_INCLUDES=(
        "val_batch*_labels.jpg"
        "val_batch*_pred.jpg"
    )

    upload_to_s3 "${SAVE_PATH}" "${S3_EVAL_PATH}" "${EVAL_INCLUDES[@]}"

    if [ $? -ne 0 ]; then
        echo -e "\e[31m[!] Evaluation results upload failed. Check s3_eval_upload.log for details.\e[0m"
        exit 1
    fi
    echo -e "\e[32m[*] Evaluation results successfully uploaded to ${S3_EVAL_PATH}.\e[0m"

    # Upload prediction files
    echo -e "${FG_BLUE}[*] Uploading prediction results...${FG_RESET}"
    upload_to_s3 "${SAVE_PATH}" "${S3_INFER_PATH}" "${PREDICTION_INCLUDES[@]}"

    if [ $? -ne 0 ]; then
        echo -e "\e[31m[!] Inference results upload failed. Check s3_infer_upload.log for details.\e[0m"
        exit 1
    fi
    echo -e "\e[32m[*] Inference results successfully uploaded to ${S3_INFER_PATH}.\e[0m"
}

# Execute steps
run_evaluate
save_results

echo -e "\e[32m[*] YOLOv5 evaluation pipeline completed successfully.\e[0m"
