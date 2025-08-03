#!/bin/bash
#
# YOLOv5 Model Evaluation Script
#
# This script automates the YOLOv5 model evaluation, including optional data/model downloads, setup, and execution.
#
# Options:
#   --model_iteration             Optional. Specify the model iteration (e.g., 1, 2). Default: "1"
#   --does_download_data_from_s3  Optional. Enable data download from S3. Default: "FALSE".
#   --does_download_model_from_s3 Optional. Enable model download from S3. Automatically enabled if `--model_iteration` is provided.
#   --skip_conda                  Optional. Skip Conda environment activation. Default: "FALSE".
#
# Usage Example:
#   ./run_yolo_v5_model_evaluation.sh --model_iteration 1 --does_download_data_from_s3 --skip_conda
#   ./run_yolo_v5_model_evaluation.sh --does_download_data_from_s3 --skip_conda
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
  --model_iteration             Optional. Specify model version or iteration (e.g., 01, 02). Default: latest local model.
  --does_download_data_from_s3  Optional. Enable data download from S3. Default: "FALSE".
  --does_download_model_from_s3 Optional. Enable model download from S3. Automatically enabled if --model_iteration is provided.
  --skip_conda                  Optional. Skip Conda environment activation. Default: "FALSE".
  -h, --help                    Show this help message and exit.
EOF
}

# Display help if requested
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    show_help
    exit 0
fi

# Default parameters
CONFIGS_DIR="${SCRIPT_DIR}/../configs"
INPUT_YAML="${CONFIGS_DIR}/input.yaml"
DOES_DOWNLOAD_DATA_FROM_S3="FALSE"
DOES_DOWNLOAD_MODEL_FROM_S3="FALSE"
MODEL_ITERATION=""
SKIP_CONDA="FALSE"
TEST_MODEL=""

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_iteration)
            MODEL_ITERATION="$2"
            DOES_DOWNLOAD_MODEL_FROM_S3="TRUE" # Automatically enable model download if iteration is specified
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
            echo -e "\e[31m[!] Error: Unknown parameter: $1\e[0m"
            show_help
            exit 1
            ;;
    esac
done

# Conda environment activation
if [[ "${SKIP_CONDA}" == "TRUE" ]]; then
    echo -e "\e[33m[!] Skipping Conda environment activation.\e[0m"
else
    update_and_activate_conda_env "${CONDA_YOLO_V5}" || {
        echo -e "\e[31m[!] Failed to activate Conda environment.\e[0m"
        exit 1
    }
fi

# Display configuration parameters
echo -e "\e[37m[*] YOLOv5 Model Evaluation Parameters:\e[0m"
echo -e "    Input YAML: ${INPUT_YAML}"
echo -e "    Model Iteration: ${MODEL_ITERATION:-1}"
echo -e "    Download Data from S3: ${DOES_DOWNLOAD_DATA_FROM_S3}"
echo -e "    Download Model from S3: ${DOES_DOWNLOAD_MODEL_FROM_S3}"
echo -e "    Skip Conda Activation: ${SKIP_CONDA}"

# Install dependencies
bash "${SCRIPT_DIR}/components/yolo_v5/install_dependencies.sh" || {
    echo -e "\e[31m[!] Dependency installation failed.\e[0m"
    exit 1
}

# Prepare inference setup
bash "${SCRIPT_DIR}/components/yolo_v5/prepare_inference_setup.sh" \
    "${INPUT_YAML}" "${DOES_DOWNLOAD_DATA_FROM_S3}" "${DOES_DOWNLOAD_MODEL_FROM_S3}" "${MODEL_ITERATION}" || {
    echo -e "\e[31m[!] Failed to prepare inference setup.\e[0m"
    exit 1
}

# Evaluate YOLOv5 model
TEST_SCRIPT="${SCRIPT_DIR}/components/yolo_v5/evaluate_model.sh"
if [[ -f "${TEST_SCRIPT}" ]]; then
    echo -e "\e[37m[*] Evaluating YOLOv5 model...\e[0m"
    bash "${TEST_SCRIPT}" "${MODEL_ITERATION}" || {
        echo -e "\e[31m[!] Model evaluation failed.\e[0m"
        exit 1
    }
else
    echo -e "\e[33m[!] Evaluation skipped. No evaluation script found.\e[0m"
fi
