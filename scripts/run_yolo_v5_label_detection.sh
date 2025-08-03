#!/bin/bash
#
# YOLOv5 Label Detection Script
#
# Automates YOLOv5 label detection, including optional data/model downloads and environment preparation.
#
# Options:
#   --model_iteration               Specify model version/iteration (e.g., 1, 2). Default: "1".
#   --does_download_data_from_s3    Enable data download from S3. Default: "FALSE".
#   --does_download_model_from_s3   Enable model download from S3. Default: "FALSE".
#   --skip_conda                    Skip Conda environment activation. Default: "FALSE".
#   -h, --help                      Show this help message.
#
# Usage:
#   ./run_yolo_v5_label_detection.sh [options]
# Example:
#   ./run_yolo_v5_label_detection.sh --model_iteration 2 --does_download_data_from_s3 --skip_conda
#

# Load environment variables and utility functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/.bin/init.sh"
source "${SCRIPT_DIR}/.bin/utils.sh"

# Help function
show_help() {
    cat << EOF
Usage: $0 [options]

Options:
  --model_iteration               Specify model version/iteration (e.g., 01, 02). Default: "1".
  --does_download_data_from_s3    Enable data download from S3. Default: "FALSE".
  --does_download_model_from_s3   Enable model download from S3. Default: "FALSE".
  --skip_conda                    Skip Conda environment activation. Default: "FALSE".
  -h, --help                      Show this help message.
EOF
}

# Show help if requested
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    show_help
    exit 0
fi

# Default parameters
CONFIGS_DIR="${SCRIPT_DIR}/../configs"
INPUT_YAML="${CONFIGS_DIR}/input.yaml"
DOES_DOWNLOAD_DATA_FROM_S3="FALSE"
DOES_DOWNLOAD_MODEL_FROM_S3="FALSE"
MODEL_ITERATION="1"
SKIP_CONDA="FALSE"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_iteration)
            MODEL_ITERATION="$2"
            DOES_DOWNLOAD_MODEL_FROM_S3="TRUE" # Enable model download if iteration specified
            shift 2
            ;;
        --does_download_data_from_s3)
            DOES_DOWNLOAD_DATA_FROM_S3="TRUE"
            shift
            ;;
        --does_download_model_from_s3)
            DOES_DOWNLOAD_MODEL_FROM_S3="TRUE"
            shift
            ;;
        --skip_conda)
            SKIP_CONDA="TRUE"
            shift
            ;;
        *)
            echo -e "${FG_RED}[!] Unknown parameter: $1${FG_RESET}"
            show_help
            exit 1
            ;;
    esac
done

# Conda environment activation
if [[ "${SKIP_CONDA}" == "TRUE" ]]; then
    echo -e "${FG_YELLOW}[!] Skipping Conda activation.${FG_RESET}"
else
    update_and_activate_conda_env "${CONDA_YOLO_V5}" || {
        echo -e "${FG_RED}[!] Failed to activate Conda environment.${FG_RESET}"
        exit 1
    }
fi

# Determine model path
if [[ "${DOES_DOWNLOAD_MODEL_FROM_S3}" == "TRUE" ]]; then
    TEST_MODEL="${LOCAL_DATA_PATH}/models/best.pt"
else
    TEST_MODEL="${PACKAGE_DIR}/yolov5/runs/train/model/weights/best.pt"
fi

# Display configuration
echo -e "${FG_GRAY}[*] YOLOv5 Label Detection Parameters:${FG_RESET}"
echo -e "${FG_GRAY}  Input YAML: ${INPUT_YAML}${FG_RESET}"
echo -e "${FG_GRAY}  Model Iteration: ${MODEL_ITERATION}${FG_RESET}"
echo -e "${FG_GRAY}  Download Data from S3: ${DOES_DOWNLOAD_DATA_FROM_S3}${FG_RESET}"
echo -e "${FG_GRAY}  Download Model from S3: ${DOES_DOWNLOAD_MODEL_FROM_S3}${FG_RESET}"
echo -e "${FG_GRAY}  Skip Conda Activation: ${SKIP_CONDA}${FG_RESET}"

# Install dependencies
bash "${SCRIPT_DIR}/components/yolo_v5/install_dependencies.sh" || {
    echo -e "${FG_RED}[!] Dependency installation failed.${FG_RESET}"
    exit 1
}

# Prepare inference setup
bash "${SCRIPT_DIR}/components/yolo_v5/prepare_inference_setup.sh" \
    "${INPUT_YAML}" "${DOES_DOWNLOAD_DATA_FROM_S3}" "${DOES_DOWNLOAD_MODEL_FROM_S3}" "${MODEL_ITERATION}" || {
    echo -e "${FG_RED}[!] Failed to prepare inference setup.${FG_RESET}"
    exit 1
}

# Run YOLOv5 label detection
LABEL_DETECTION_SCRIPT="${SCRIPT_DIR}/components/yolo_v5/detect_labels.sh"
if [[ -f "${LABEL_DETECTION_SCRIPT}" ]]; then
    echo -e "${FG_GRAY}[*] Running YOLOv5 label detection...${FG_RESET}"
    bash "${LABEL_DETECTION_SCRIPT}" "${MODEL_ITERATION}"|| {
        echo -e "${FG_RED}[!] Label detection failed.${FG_RESET}"
        exit 1
    }
else
    echo -e "${FG_YELLOW}[!] Skipping label detection. Script not found: ${LABEL_DETECTION_SCRIPT}.${FG_RESET}"
fi
