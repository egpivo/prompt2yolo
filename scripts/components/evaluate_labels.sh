#!/bin/bash

# Label Evaluation Script
# Automates the label evaluation and optionally uploads results to S3.
#
# Arguments:
#   $1 - INPUT_YAML (required): Path to the input YAML file containing prompts and classes.
#   $2 - ITERATION (optional): Iteration number to organize outputs by iteration folders. Defaults to 1.
#
# Usage:
#   bash evaluate_label.sh ../configs/input.yaml 1

# Load environment variables, utility functions, and color map
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="${CURRENT_DIR}/../.."
source "${PACKAGE_DIR}/scripts/.bin/init.sh"
source "${PACKAGE_DIR}/scripts/.bin/utils.sh"
source "${PACKAGE_DIR}/scripts/.bin/color_map.sh"

# Parse and assign arguments
INPUT_YAML="$1"
ITERATION="${2:-1}"

# Validate required arguments
if [ -z "${INPUT_YAML}" ]; then
    echo -e "${FG_RED}[!] Error: Missing required argument: INPUT_YAML.${FG_RESET}"
    echo -e "Usage: $0 <INPUT_YAML> [ITERATION]"
    exit 1
fi

# Set up directories
LOCAL_DATA_PATH="${LOCAL_DATA_PATH:-local_temp}"  # Default fallback
EVALUATION_OUTPUT_DIR="${LOCAL_DATA_PATH}/yolo/data/iteration_${ITERATION}/test"

# Ensure the output directory exists
mkdir -p "${EVALUATION_OUTPUT_DIR}"

# Function to run label evaluation
run_evaluation() {
    echo -e "${FG_BLUE}[*] Running label evaluation for iteration ${ITERATION}...${FG_RESET}"

    python "${PACKAGE_DIR}/prompt2yolo/execution/run_label_evaluation.py" \
        --input_yaml "${INPUT_YAML}" \
        --iteration "${ITERATION}"

    if [ $? -eq 0 ]; then
        echo -e "${FG_GREEN}[*] Label evaluation completed successfully for iteration ${ITERATION}.${FG_RESET}"
    else
        echo -e "${FG_RED}[!] Label evaluation failed for iteration ${ITERATION}.${FG_RESET}"
        exit 1
    fi
}

# Function to upload results to S3
save_results() {
    echo -e "${FG_BLUE}[*] Saving evaluation results for iteration ${ITERATION} to S3...${FG_RESET}"

    # Define the S3 base path for uploading results
    S3_BASE_PATH="s3://${AWS_S3_BUCKET_NAME}/projects/${PROJECT}/models/iteration_${ITERATION}/evaluation/label"

    # Check if the evaluation output directory exists
    if [ ! -d "${EVALUATION_OUTPUT_DIR}" ]; then
        echo -e "${FG_RED}[!] Error: Output directory '${EVALUATION_OUTPUT_DIR}' does not exist.${FG_RESET}"
        exit 1
    fi

    # Define directories for the three categories
    FALSE_NEGATIVE_DIR="${EVALUATION_OUTPUT_DIR}/separation_results/false_negative"
    FALSE_POSITIVE_DIR="${EVALUATION_OUTPUT_DIR}/separation_results/false_positive"
    TRUE_POSITIVE_DIR="${EVALUATION_OUTPUT_DIR}/separation_results/true_positive"

    # Ensure each directory exists before uploading
    for CATEGORY_DIR in "${FALSE_NEGATIVE_DIR}" "${FALSE_POSITIVE_DIR}" "${TRUE_POSITIVE_DIR}"; do
        if [ ! -d "${CATEGORY_DIR}" ]; then
            echo -e "${FG_YELLOW}[!] Warning: Directory '${CATEGORY_DIR}' does not exist. Skipping this category.${FG_RESET}"
            continue
        fi

        # Construct the S3 path for the current category
        CATEGORY_NAME=$(basename "${CATEGORY_DIR}")
        S3_PATH="${S3_BASE_PATH}/category/${CATEGORY_NAME}"

        # Upload files to S3
        echo -e "${FG_BLUE}[*] Uploading ${CATEGORY_NAME} results to ${S3_PATH}...${FG_RESET}"
        upload_to_s3 "${CATEGORY_DIR}" "${S3_PATH}" "*.png" "*.jpg"

        if [ $? -eq 0 ]; then
            echo -e "${FG_GREEN}[*] Successfully uploaded ${CATEGORY_NAME} results to ${S3_PATH}.${FG_RESET}"
        else
            echo -e "${FG_RED}[!] Failed to upload ${CATEGORY_NAME} results to S3. Check logs for details.${FG_RESET}"
        fi
    done

    # Upload individual plot files to S3
    PLOTS_DIR="${EVALUATION_OUTPUT_DIR}/separation_results"
    PLOTS=("label_prompt_fp_rate_distribution.jpg" "label_prompt_weight.jpg")
    S3_PLOT_PATH="${S3_BASE_PATH}/statistics"

    for PLOT in "${PLOTS[@]}"; do
        PLOT_PATH="${PLOTS_DIR}/${PLOT}"

        if [ ! -f "${PLOT_PATH}" ]; then
            echo -e "${FG_YELLOW}[!] Warning: Plot '${PLOT}' does not exist in '${PLOTS_DIR}'. Skipping this plot.${FG_RESET}"
            continue
        fi

        # Directly upload the file without recursive mode
        echo -e "${FG_BLUE}[*] Uploading plot '${PLOT}' to ${S3_PLOT_PATH}...${FG_RESET}"
        aws s3 cp "${PLOT_PATH}" "${S3_PLOT_PATH}/${PLOT}" --endpoint-url "${AWS_S3_ENDPOINT}" || {
            echo -e "${FG_RED}[!] Failed to upload plot '${PLOT}' to S3.${FG_RESET}"
            exit 1
        }

        echo -e "${FG_GREEN}[*] Successfully uploaded plot '${PLOT}' to ${S3_PLOT_PATH}.${FG_RESET}"
    done

    echo -e "${FG_GREEN}[*] All categories and plots processed for iteration ${ITERATION}.${FG_RESET}"
}

# Execute the pipeline
run_evaluation
save_results

echo -e "${FG_GREEN}[*] Label evaluation pipeline for iteration ${ITERATION} completed successfully.${FG_RESET}"
