#!/bin/bash
#
# Human Detection Model Training Pipeline
#
# Automates the YOLO-based pipeline for human detection, including data generation, model training,
# inference, and evaluation. Iteratively checks convergence based on the False Positive (FP) rate.
#
# Parameters:
#   --model          Optional. YOLO model version to use. Default: `yolo_v5`. Options: `yolo_v5`, `yolo_v3_tiny`.
#   --skip_conda     Optional. Skip Conda environment activation. Default: `FALSE`.
#   --max_iterations Optional. Maximum number of iterations for the retraining loop. Default: 5.
#   --convergence_threshold Optional. Threshold for convergence. Default: 0.1.
#
# Usage Example:
#   ./automate_human_detection_model_training.sh --model yolo_v5 --skip_conda --max_iterations 10
#
# Outputs:
#   - Model weights, detection results, evaluation metrics, and FP rate logs.
#

# Load environment variables, utility functions, and color map
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/.bin/init.sh"
source "${CURRENT_DIR}/.bin/utils.sh"
source "${CURRENT_DIR}/.bin/color_map.sh"
configure_aws_from_env

# Default configurations
CONFIGS_DIR="${CURRENT_DIR}/../configs"
MODEL="yolo_v5"
SKIP_CONDA="FALSE"
INPUT_YAML="${CONFIGS_DIR}/input.yaml"
MAX_ITERATIONS=5
CONVERGENCE_THRESHOLD=0.1
LOCAL_DATA_PATH=$(echo "${LOCAL_DATA_PATH}" | sed 's/^"//;s/"$//') # Handle quotes in env variables



# Show help function
show_help() {
    cat << EOF
Usage: $0 [options]

Options:
  --model                 Specify the model type ('yolo_v5' or 'yolo_v3_tiny'). Default: 'yolo_v5'.
  --skip_conda            Skip Conda environment activation.
  --max_iterations        Maximum number of iterations for the retraining loop. Default: 5.
  --convergence_threshold Threshold for convergence based on False Positive rate. Default: 0.1.
  --help, -h              Display this help message.
EOF
}


# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --skip_conda)
            SKIP_CONDA="TRUE"
            shift
            ;;
        --max_iterations)
            MAX_ITERATIONS="$2"
            if ! echo "$MAX_ITERATIONS" | grep -qE '^[0-9]+$' || [ "$MAX_ITERATIONS" -le 0 ]; then
                echo "[!] Error: --max_iterations must be a positive integer."
                exit 1
            fi
            shift 2
            ;;
        --convergence_threshold)
            CONVERGENCE_THRESHOLD="$2"
            if ! echo "$CONVERGENCE_THRESHOLD" | grep -qE '^[0-9]+(\.[0-9]+)?$' || (( $(echo "$CONVERGENCE_THRESHOLD < 0" | bc -l) )); then
                echo "[!] Error: --convergence_threshold must be a non-negative number."
                exit 1
            fi
            shift 2
            ;;
        *)
            echo -e "[!] Error: Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# Set Conda environment
CONDA_ENV_NAME="${CONDA_YOLO_V5}"
if [[ "${MODEL}" == "yolo_v3_tiny" ]]; then
    CONDA_ENV_NAME="${CONDA_YOLO_V3_TINY}"
fi

# Activate Conda unless skipped
if [[ "${SKIP_CONDA}" == "TRUE" ]]; then
    echo -e "${FG_YELLOW}[!] Skipping Conda environment activation.${FG_RESET}"
    pip install .
else
    update_and_activate_conda_env "${CONDA_ENV_NAME}" || {
        echo -e "${FG_RED}[!] Failed to activate Conda environment: ${CONDA_ENV_NAME}${FG_RESET}"
        exit 1
    }
fi

# Function to run pipeline steps
run_step() {
    local script_path="$1"
    local step_name="$2"
    shift 2

    if [[ -f "${script_path}" ]]; then
        echo -e "${FG_BLUE}[*] Running ${step_name}...${FG_RESET}"
        bash "${script_path}" "$@" || {
            echo -e "${FG_RED}[!] ${step_name} failed. Stopping pipeline.${FG_RESET}"
            exit 1
        }
    else
        echo -e "${FG_YELLOW}[!] Skipping ${step_name}. Script not found: ${script_path}${FG_RESET}"
    fi
}

# Preparation steps
run_step "${CURRENT_DIR}/components/${MODEL}/install_dependencies.sh" "Dependency Installation"

# Iterative retraining loop
CURRENT_ITERATION=1

while [[ "${CURRENT_ITERATION}" -le "${MAX_ITERATIONS}" ]]; do
    echo -e "${FG_BLUE}[*] Iteration ${CURRENT_ITERATION}...${FG_RESET}"

    # Data generation
    run_step "${CURRENT_DIR}/components/generate_image_data.sh" \
        "Data Generation" "${INPUT_YAML}" "${CONFIGS_DIR}/${MODEL}/image_generator.yaml" "${CONFIGS_DIR}/${MODEL}/image_labeler.yaml" "${CURRENT_ITERATION}"

    # Model training
    run_step "${CURRENT_DIR}/components/${MODEL}/prepare_training_setup.sh" "Training Setup" "${INPUT_YAML}" "${S3_MODEL_NAME}" "${CURRENT_ITERATION}"
    run_step "${CURRENT_DIR}/components/${MODEL}/train_model.sh" "Model Training" "${CURRENT_ITERATION}"

    # Evaluation and label evaluation
    run_step "${CURRENT_DIR}/components/${MODEL}/prepare_inference_setup.sh" "Inference Preparation" "${INPUT_YAML}" "FALSE" "TRUE" "${CURRENT_ITERATION}"
    run_step "${CURRENT_DIR}/components/${MODEL}/evaluate_model.sh" \
        "Model Evaluation" "${CURRENT_ITERATION}"
    run_step "${CURRENT_DIR}/components/${MODEL}/detect_labels.sh" \
        "Label Detection" "${CURRENT_ITERATION}"
    run_step "${CURRENT_DIR}/components/evaluate_labels.sh" \
        "Label Evaluation" "${INPUT_YAML}" "${CURRENT_ITERATION}"

    # Check convergence using FP rate
    FP_RATE_DIR="${LOCAL_DATA_PATH}/yolo/data/iteration_${CURRENT_ITERATION}/test/separation_results"
    FP_RATE_FILE="${FP_RATE_DIR}/max_fp_rate.txt"

    if [[ ! -f "${FP_RATE_FILE}" ]]; then
        echo -e "${FG_RED}[!] FP rate file not found: ${FP_RATE_FILE}. Stopping iteration.${FG_RESET}"
        break
    fi

    MAX_FP_RATE=$(cat "${FP_RATE_FILE}")
    if [[ -z "${MAX_FP_RATE}" ]]; then
        echo -e "${FG_RED}[!] FP rate file is empty: ${FP_RATE_FILE}. Stopping iteration.${FG_RESET}"
        break
    fi

    echo -e "${FG_BLUE}[*] Max False Positive Rate: ${MAX_FP_RATE}${FG_RESET}"
    if (( $(echo "${MAX_FP_RATE} < ${CONVERGENCE_THRESHOLD}" | bc -l) )); then
        echo -e "${FG_GREEN}[*] Convergence achieved. Stopping pipeline.${FG_RESET}"
        break
    fi

    ((CURRENT_ITERATION++))
done

echo -e "${FG_GREEN}[*] Pipeline execution with iterative retraining completed.${FG_RESET}"
