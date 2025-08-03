#!/bin/bash
# YOLOv5 Dependency Installation Script
# This script installs necessary dependencies and clones the YOLOv5 repository from GitHub.
#
# Description:
#   - Loads required environment variables.
#   - Checks if the YOLOv5 repository is already present in the current directory.
#   - If not, clones the YOLOv5 repository from GitHub.
#   - Navigates to the cloned repository, fetches the latest changes, and checks out a specific commit.
#   - Installs Python dependencies listed in the `requirements.txt` file of the repository.
#   - Returns to the original directory after completing the installation.
#
# Prerequisites:
#   - Ensure that Python and Git are installed on your system.
#   - Make sure required environment variables are set in the init script (`init.sh`).
#
# Usage:
#   ./install_dependencies.sh
#

# Load environment variables
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="${CURRENT_DIR}/../../.."

source "${PACKAGE_DIR}/scripts/.bin/init.sh"

# Function to install dependencies and clone YOLOv5 repository
install_dependencies() {
    echo -e "${FG_GRAY}[*] Installing dependencies and cloning YOLOv5 repository...${FG_RESET}"

    if [ -d "${PACKAGE_DIR}/yolov5" ]; then
        echo -e "${FG_YELLOW}[!] YOLOv5 repository already exists. Removing it before cloning.${FG_RESET}"
        rm -rf "${PACKAGE_DIR}/yolov5" || {
            echo -e "${FG_RED}[!] Failed to remove existing YOLOv5 repository.${FG_RESET}"
            exit 1
        }
    fi

    git clone https://github.com/ultralytics/yolov5 "${PACKAGE_DIR}/yolov5" || {
        echo -e "${FG_RED}[!] Failed to clone YOLOv5 repository.${FG_RESET}"
        exit 1
    }
    # Navigate into the yolov5 directory
    pushd "${PACKAGE_DIR}/yolov5" > /dev/null || {
        echo -e "${FG_RED}[!] Failed to access yolov5 directory.${FG_RESET}"
        exit 1
    }

    # Fetch the latest changes
    git fetch --all || {
        echo -e "${FG_RED}[!] Failed to fetch updates from YOLOv5 repository.${FG_RESET}"
        exit 1
    }

    # Checkout a specific commit to ensure the correct version
    COMMIT_HASH="24ee28010fbf597ec796e6e471429cde21040f90"
    git checkout "${COMMIT_HASH}" || {
        echo -e "${FG_RED}[!] Failed to checkout specified commit: ${COMMIT_HASH}.${FG_RESET}"
        exit 1
    }

    # Install dependencies listed in requirements.txt
    echo -e "${FG_GRAY}[*] Installing Python dependencies...${FG_RESET}"
    pip install -qr requirements.txt || {
        echo -e "${FG_RED}[!] Failed to install YOLOv5 dependencies.${FG_RESET}"
        exit 1
    }

    # Return to the original directory
    popd > /dev/null || {
        echo -e "${FG_RED}[!] Failed to return to the original directory.${FG_RESET}"
        exit 1
    }

    echo -e "${FG_GREEN}[*] YOLOv5 dependencies installed successfully.${FG_RESET}"
}

# Execute the installation function
install_dependencies
