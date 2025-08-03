#!/bin/bash

# YOLOv3-Tiny Installation Script
# This script installs dependencies, clones the Darknet repository, checks out a specific commit,
# updates the Makefile to enable OpenCV, GPU, and CUDNN support, and downloads pre-trained weights.
#
# Arguments:
#   $1 - Number of GPUs (optional): Specifies the number of GPUs to use. Defaults to 1 if not provided.
#
# Usage Example:
#   ./install_dependencies.sh           # Uses 1 GPU by default
#   ./install_dependencies.sh 2         # Uses 2 GPUs

# Get the directory of the script
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="${CURRENT_DIR}/../../.."
source "${PACKAGE_DIR}/scripts/.bin/init.sh"

# Validate the number of parameters passed
if [ "$#" -gt 1 ]; then
    echo "Usage: $0 [number_of_gpus]"
    exit 1
fi

# Set the default number of GPUs to 1 if no argument is provided
NUM_GPUS="${1:-1}"

# Install dependencies and clone the Darknet repository
install_dependencies() {
    echo -e "${FG_BLUE}[*] Updating package lists...${FG_RESET}"
    sudo apt update

    echo -e "${FG_BLUE}[*] Installing OpenCV and CUDA toolkit...${FG_RESET}"
    sudo apt install -y libopencv-dev nvidia-cuda-toolkit || {
        echo -e "${FG_RED}[!] Failed to install OpenCV or CUDA toolkit.${FG_RESET}"
        exit 1
    }

    # Check for CUDA installation
    if [ ! -d "/usr/local/cuda" ]; then
        echo -e "${FG_RED}[!] CUDA not found. Please install the CUDA toolkit manually from NVIDIA's site.${FG_RESET}"
        exit 1
    else
        echo -e "${FG_GREEN}[✓] CUDA found.${FG_RESET}"
    fi

    # Detect CUDA version
    CUDA_VERSION=$(nvcc --version | grep -o "release [0-9]*" | awk '{print $2}' | cut -d '.' -f1)
    echo -e "${FG_BLUE}[*] Detected CUDA version: ${CUDA_VERSION}${FG_RESET}"

    # Install the appropriate cuDNN version based on CUDA version
    if [ "${CUDA_VERSION}" == "11" ]; then
        echo -e "${FG_BLUE}[*] Installing cuDNN for CUDA 11...${FG_RESET}"
        sudo apt-get -y install cudnn9-cuda-11 || {
            echo -e "${FG_RED}[!] Failed to install cuDNN for CUDA 11. Please check compatibility manually.${FG_RESET}"
            exit 1
        }
    elif [ "${CUDA_VERSION}" == "12" ]; then
        echo -e "${FG_BLUE}[*] Installing cuDNN for CUDA 12...${FG_RESET}"
        sudo apt-get -y install cudnn9-cuda-12 || {
            echo -e "${FG_RED}[!] Failed to install cuDNN for CUDA 12. Please check compatibility manually.${FG_RESET}"
            exit 1
        }
    else
        echo -e "${FG_RED}[!] Unsupported CUDA version detected: ${CUDA_VERSION}. Please install the appropriate cuDNN version manually.${FG_RESET}"
        exit 1
    fi

    echo -e "${FG_GREEN}[✓] cuDNN installed successfully.${FG_RESET}"

    # Clone Darknet repository if it doesn't exist
    if [ ! -d "darknet" ]; then
        git clone https://github.com/AlexeyAB/darknet.git || {
            echo -e "${FG_RED}[!] Failed to clone Darknet repository.${FG_RESET}"
            exit 1
        }
    fi

    # Navigate into the Darknet directory
    pushd darknet > /dev/null || {
        echo -e "${FG_RED}[!] Failed to access darknet directory.${FG_RESET}"
        exit 1
    }

    # Fetch the latest changes from the repository
    git fetch --all

    # Checkout the specific commit to ensure version compatibility
    COMMIT_HASH="19dde2f296941a75b0b9202cccd59528bde7f65a"
    echo -e "${FG_GRAY}[*] Checking out specific commit: ${COMMIT_HASH}...${FG_RESET}"
    git checkout "${COMMIT_HASH}" || {
        echo -e "${FG_RED}[!] Failed to checkout specified commit.${FG_RESET}"
        exit 1
    }

    # Update Makefile to enable OpenCV, GPU, and CUDNN support
    echo -e "${FG_GRAY}[*] Updating Makefile to enable OpenCV, GPU, and CUDNN...${FG_RESET}"
    sed -i 's/OPENCV=0/OPENCV=1/' Makefile
    sed -i "s/GPU=0/GPU=${NUM_GPUS}/" Makefile
    sed -i 's/CUDNN=0/CUDNN=1/' Makefile

    # Compile Darknet
    echo -e "${FG_GRAY}[*] Compiling Darknet...${FG_RESET}"
    make || {
        echo -e "${FG_RED}[!] Failed to compile Darknet.${FG_RESET}"
        exit 1
    }

    # Download YOLOv3-Tiny pre-trained weights
    echo -e "${FG_GRAY}[*] Downloading YOLOv3-Tiny pre-trained weights...${FG_RESET}"
    wget https://pjreddie.com/media/files/yolov3-tiny.weights -O yolov3-tiny.weights || {
        echo -e "${FG_RED}[!] Failed to download YOLOv3-Tiny weights.${FG_RESET}"
        exit 1
    }

    # Perform partial conversion of YOLOv3-Tiny weights
    echo -e "${FG_GRAY}[*] Performing partial conversion of YOLOv3-Tiny weights...${FG_RESET}"
    ./darknet partial cfg/yolov3-tiny.cfg yolov3-tiny.weights yolov3-tiny.conv.15 15 || {
        echo -e "${FG_RED}[!] Failed to perform partial conversion of YOLOv3-Tiny weights.${FG_RESET}"
        exit 1
    }

    # Return to the original directory
    popd > /dev/null

    echo -e "${FG_GREEN}[*] Dependencies installed successfully.${FG_RESET}"
}

# Call the install_dependencies function
install_dependencies
