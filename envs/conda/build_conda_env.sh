#!/bin/bash
#
# Build Conda Environment
#
# This script builds a Conda environment based on the provided name or the default 'diffusion'.
#
# Parameters:
#   -c, --conda_env       Specifies the name of the Conda environment (default: 'diffusion')
#
# Examples:
#   1. Build the default Conda environment 'prompt2yolo':
#       ./build_conda_env.sh
#
#   2. Build a specific Conda environment 'my_env':
#       ./build_conda_env.sh -c my_env
#
# Note:
#   - If 'realpath' is not available on macOS, install it via 'brew install coreutils'.
#

# Force the script to run in bash
if [ -z "$BASH_VERSION" ]; then
    echo "This script requires bash. Please run it with bash: bash $0"
    exit 1
fi

# Directory of the script
CONDA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set -a

# Ensure files exist before sourcing
if [ -f "${CONDA_DIR}/conda_env_info.sh" ]; then
    source "${CONDA_DIR}/conda_env_info.sh"
else
    echo "Error: conda_env_info.sh not found in ${CONDA_DIR}"
    exit 1
fi

if [ -f "${CONDA_DIR}/utils.sh" ]; then
    source "${CONDA_DIR}/utils.sh"
else
    echo "Error: utils.sh not found in ${CONDA_DIR}"
    exit 1
fi

if [ -f "${COLOR_MAP_PATH}" ]; then
    source "${COLOR_MAP_PATH}"
else
    echo "Error: COLOR_MAP_PATH not found"
    exit 1
fi

if [ -f "${EXIT_CODE_PATH}" ]; then
    source "${EXIT_CODE_PATH}"
else
    echo "Error: EXIT_CODE_PATH not found"
    exit 1
fi

set +a

# Default values for options
CONDA_ENV="prompt2yolo"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--conda_env)
            CONDA_ENV="$2"
            shift 2
            ;;
        *)
            echo "Invalid option: $1"
            exit 1
            ;;
    esac
done

# Function to build the Conda environment
build() {
    local CONDA_ENV="$1"

    # Check if the Conda environment exists
    echo -e "${FG_YELLOW}Checking if Conda environment '${CONDA_ENV}' exists...${FG_RESET}"
    if ! CONDA_ENV_DIR="$(find_conda_env_path "${CONDA_ENV}")"; then
        echo -e "${FG_YELLOW}Environment not found. Attempting to create with Python version '${DEFAULT_PYTHON_VERSION}'...${FG_RESET}"
        if ! retry_to_find_conda_env_path "${CONDA_ENV}" "${DEFAULT_PYTHON_VERSION}"; then
            echo -e "${FG_RED}[!] Error: Conda environment '${CONDA_ENV}' not found and could not be created.${FG_RESET}"
            return 1
        fi
    fi

    # Initialize Conda
    echo -e "${FG_YELLOW}Initializing Conda...${FG_RESET}"
    if ! initialize_conda; then
        echo -e "${FG_RED}[!] Error: Failed to initialize Conda.${FG_RESET}"
        return 1
    fi

    # Activate the Conda environment
    conda activate $CONDA_ENV
    echo -e "${FG_GREEN}Conda environment '${CONDA_ENV}' activated.${FG_RESET}"

    # Install packages
    echo -e "${FG_YELLOW}Installing required packages...${FG_RESET}"
    if ! install_python_package "${PACKAGE_BASE_PATH}"; then
        echo -e "${FG_RED}[!] Error: Failed to install required packages.${FG_RESET}"
        return 1
    fi
    echo -e "${FG_GREEN}Packages installed successfully.${FG_RESET}"

    # Export environment configuration
    echo -e "${FG_YELLOW}Exporting Conda environment configuration to 'environment.yml'...${FG_RESET}"
    conda env export > "${CONDA_DIR}/environment.yml"
    if [[ $? -ne 0 ]]; then
        echo -e "${FG_RED}[!] Error: Failed to export environment configuration.${FG_RESET}"
        return 1
    fi
    echo -e "${FG_GREEN}Environment configuration exported successfully.${FG_RESET}"

    # Deactivate the Conda environment
    echo -e "${FG_YELLOW}Deactivating Conda environment '${CONDA_ENV}'...${FG_RESET}"
    conda deactivate
    if [[ $? -ne 0 ]]; then
        echo -e "${FG_RED}[!] Error: Failed to deactivate Conda environment.${FG_RESET}"
        return 1
    fi
    echo -e "${FG_GREEN}Conda environment '${CONDA_ENV}' deactivated.${FG_RESET}"

    return 0
}

# Build the Conda environment
build "${CONDA_ENV}"
