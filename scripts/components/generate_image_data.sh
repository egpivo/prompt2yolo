#!/bin/bash
# Data Generation Script with Upload to S3
#
# Arguments:
#   $1 - INPUT_YAML (required): Path to the YAML file containing input data and configuration details.
#   $2 - IMAGE_GENERATOR_YAML (required): Path to the YAML file for the image generator configuration.
#   $3 - IMAGE_LABELER_YAML (required): Path to the YAML file for the image labeler configuration.
#   $4 - ITERATION (optional): Iteration number, defaults to 1.
#
# Usage Example:
#   ./generate_image_data.sh ../configs/input.yaml ../configs/image_generator.yaml ../configs/image_labeler.yaml 1
#
# Description:
#   - Runs data generation and uploads generated datasets (train, val, test) to S3.
#

# Load environment variables and utility functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../.bin/init.sh"
source "${SCRIPT_DIR}/../.bin/color_map.sh"

# Parse arguments
INPUT_YAML="$1"
IMAGE_GENERATOR_YAML="$2"
IMAGE_LABELER_YAML="$3"
ITERATION="${4:-1}"

# Validate arguments
if [ -z "${INPUT_YAML}" ] || [ -z "${IMAGE_GENERATOR_YAML}" ] || [ -z "${IMAGE_LABELER_YAML}" ]; then
    echo -e "${FG_RED}[!] Error: Missing arguments.${FG_RESET}"
    echo -e "${FG_YELLOW}Usage: $0 <input_yaml> <image_generator_yaml> <image_labeler_yaml> [iteration]${FG_RESET}"
    exit 1
fi

# Function: Run data generation
generate_data() {
    echo -e "${FG_BLUE}[*] Running data generation for iteration ${ITERATION}...${FG_RESET}"
    python "${PACKAGE_DIR}/prompt2yolo/execution/run_data_generation.py" \
        --input_yaml "${INPUT_YAML}" \
        --image_generator_yaml "${IMAGE_GENERATOR_YAML}" \
        --image_labeler_yaml "${IMAGE_LABELER_YAML}" \
        --iteration "${ITERATION}"

    if [ $? -eq 0 ]; then
        echo -e "${FG_GREEN}[*] Data generation completed successfully for iteration ${ITERATION}.${FG_RESET}"
    else
        echo -e "${FG_RED}[!] Data generation failed for iteration ${ITERATION}.${FG_RESET}"
        exit 1
    fi
}

# Function: Upload dataset to S3
upload_dataset() {
    local LOCAL_FOLDER=$1
    local S3_FOLDER=$2

    echo -e "${FG_BLUE}[*] Uploading from ${LOCAL_FOLDER} to s3://${AWS_S3_BUCKET_NAME}/${S3_FOLDER}...${FG_RESET}"
    aws s3 cp "${LOCAL_FOLDER}" "s3://${AWS_S3_BUCKET_NAME}/${S3_FOLDER}" --recursive --endpoint-url "${AWS_S3_ENDPOINT}"

    if [ $? -eq 0 ]; then
        echo -e "${FG_GREEN}[*] Successfully uploaded to s3://${AWS_S3_BUCKET_NAME}/${S3_FOLDER}.${FG_RESET}"
    else
        echo -e "${FG_RED}[!] Failed to upload to s3://${AWS_S3_BUCKET_NAME}/${S3_FOLDER}.${FG_RESET}"
        exit 1
    fi
}

upload_all_datasets() {
    echo -e "${FG_YELLOW}[*] Uploading datasets for iteration ${ITERATION}...${FG_RESET}"

    for MODE in "train" "val" "test"; do
        for TYPE in "images" "labels"; do
            LOCAL_FOLDER="${LOCAL_DATA_PATH}/yolo/data/iteration_${ITERATION}/${MODE}/${TYPE}"
            S3_FOLDER="projects/${PROJECT}/data/iteration_${ITERATION}/${MODE}/${TYPE}"

            if [ -d "${LOCAL_FOLDER}" ]; then
                upload_dataset "${LOCAL_FOLDER}" "${S3_FOLDER}"
            else
                echo -e "${FG_RED}[!] Missing ${MODE}/${TYPE} folder. Skipping upload.${FG_RESET}"
            fi
        done
    done

    echo -e "${FG_GREEN}[*] Dataset upload completed.${FG_RESET}"
}

# Execute pipeline
generate_data
upload_all_datasets
