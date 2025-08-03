#!/bin/bash
#
# YOLOv5 Model Training and Upload Script
# This script automates the setup, training, and saving of a YOLOv5 model for object detection.
#
# Arguments:
#   $1 - ITERATION (optional): Iteration number to include in paths. Defaults to 1.
#
# Usage Example:
#   ./train_model.sh 1
#

# Arguments
ITERATION="${1:-1}"

# Load environment variables
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="${CURRENT_DIR}/../../.."
source "${PACKAGE_DIR}/scripts/.bin/init.sh"
source "${PACKAGE_DIR}/scripts/.bin/utils.sh"

# Load training configuration
TRAINING_CONFIG_FILE="${PACKAGE_DIR}/configs/yolo_v5/training.yaml"
TRAINING_DATA_CONFIG=$(get_yaml_value "data_config" "${TRAINING_CONFIG_FILE}")
HYP_CONFIG=$(get_yaml_value "hyp_config" "${TRAINING_CONFIG_FILE}")
OPTIMIZER=$(get_yaml_value "optimizer" "${TRAINING_CONFIG_FILE}")
PATIENCE=$(get_yaml_value "patience" "${TRAINING_CONFIG_FILE}")
EPOCHS=$(get_yaml_value "epochs" "${TRAINING_CONFIG_FILE}")
BATCH_SIZE=$(get_yaml_value "batch_size" "${TRAINING_CONFIG_FILE}")
YOLO_MODEL=$(get_yaml_value "yolo_model" "${TRAINING_CONFIG_FILE}")
TRAINING_DIR_NAME="$(get_yaml_value "directory" "${TRAINING_CONFIG_FILE}")/iteration_${ITERATION}"

# Determine weights
if [[ "${ITERATION}" -eq 1 ]]; then
    # Use original YOLO weights for the first iteration
    WEIGHTS_PATH="${YOLO_MODEL}"
else
    # Use the weights from the previous iteration
    PREVIOUS_ITERATION=$((ITERATION - 1))
    PREVIOUS_TRAINING_DIR_NAME="$(get_yaml_value "directory" "${TRAINING_CONFIG_FILE}")/iteration_${PREVIOUS_ITERATION}"
    WEIGHTS_PATH="${PACKAGE_DIR}/yolov5/runs/train/${PREVIOUS_TRAINING_DIR_NAME}/weights/best.pt"

    if [[ ! -f "${WEIGHTS_PATH}" ]]; then
        echo -e "${FG_RED}[!] Previous weights not found at ${WEIGHTS_PATH}. Falling back to initial YOLO weights.${FG_RESET}"
        WEIGHTS_PATH="${YOLO_MODEL}"
    fi
fi

# Resolve absolute paths
HYP_CONFIG_PATH=$(readlink -f "${LOCAL_DATA_PATH}/yolo/configs/${HYP_CONFIG}")
DATA_CONFIG_PATH=$(readlink -f "${LOCAL_DATA_PATH}/yolo/configs/${TRAINING_DATA_CONFIG}")
SAVE_PATH="${PACKAGE_DIR}/yolov5/runs/train/${TRAINING_DIR_NAME}/"

# Clean destination folder
clean_destination() {
    echo -e "${FG_YELLOW}[*] Cleaning destination folder: ${SAVE_PATH}${FG_RESET}"
    rm -rf "${SAVE_PATH}"
}

# Train YOLOv5
train_yolo() {
    echo -e "${FG_BLUE}[*] Starting YOLOv5 training...${FG_RESET}"
    python "${PACKAGE_DIR}/yolov5/train.py" \
      --img 416 \
      --batch "${BATCH_SIZE}" \
      --epochs "${EPOCHS}" \
      --patience "${PATIENCE}" \
      --hyp "${HYP_CONFIG_PATH}" \
      --data "${DATA_CONFIG_PATH}" \
      --optimizer "${OPTIMIZER}" \
      --weights "${WEIGHTS_PATH}" \
      --name "${TRAINING_DIR_NAME}"

    if [ $? -eq 0 ]; then
        echo -e "${FG_GREEN}[*] Training completed successfully.${FG_RESET}"
    else
        echo -e "${FG_RED}[!] Training failed.${FG_RESET}"
        exit 1
    fi
}

# Upload trained model to S3
upload_model() {
    echo -e "${FG_BLUE}[*] Uploading model to S3...${FG_RESET}"

    # Define paths
    S3_EVAL_PATH="s3://${AWS_S3_BUCKET_NAME}/projects/${PROJECT}/models/iteration_${ITERATION}/evaluation/model/training/performance"
    S3_PREDICTION_PATH="s3://${AWS_S3_BUCKET_NAME}/projects/${PROJECT}/models/iteration_${ITERATION}/evaluation/model/training/predictions"
    S3_WEIGHTS_PATH="s3://${AWS_S3_BUCKET_NAME}/projects/${PROJECT}/models/iteration_${ITERATION}"

    # Define inclusion lists
    EVAL_INCLUDES=(
        "*_curve.png"
        "confusion_matrix.png"
        "labels_correlogram.jpg"
        "results.png"
    )
    PREDICTION_INCLUDES=(
        "val_batch*_labels.jpg"
        "val_batch*_pred.jpg"
        "train_batch*.jpg"
    )
    WEIGHTS_INCLUDES=(
        "weights/*.pt"
    )

    # Upload evaluation files
    echo -e "${FG_BLUE}[*] Uploading evaluation results...${FG_RESET}"
    upload_to_s3 "${SAVE_PATH}" "${S3_EVAL_PATH}" "${EVAL_INCLUDES[@]}"

    # Upload prediction files
    echo -e "${FG_BLUE}[*] Uploading prediction results...${FG_RESET}"
    upload_to_s3 "${SAVE_PATH}" "${S3_PREDICTION_PATH}" "${PREDICTION_INCLUDES[@]}"

    # Upload weights
    echo -e "${FG_BLUE}[*] Uploading weights...${FG_RESET}"
    upload_to_s3 "${SAVE_PATH}" "${S3_WEIGHTS_PATH}" "${WEIGHTS_INCLUDES[@]}"

    echo -e "${FG_GREEN}[*] All files successfully uploaded to S3.${FG_RESET}"
}

# Execute functions
clean_destination
train_yolo
upload_model
