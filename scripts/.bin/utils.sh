#!/bin/bash

# Source color definitions
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="${DIR}/../.."
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"/color_map.sh

# Function to extract values from a YAML file
get_yaml_value() {
  local key="$1"
  local file="$2"
  awk -F": " "/^$key:/ {print \$2}" "$file" | tr -d '"'  # Removes any quotes
}

# Function to display help information
show_help() {
    sed -n '2,/^$/p' "$0"
}

# Function to generate a new model version based on timestamp
get_new_model_version() {
  date +"%Y%m%d%H%M%S"
}

# Function to update and activate a specified Conda environment
update_and_activate_conda_env() {
  local conda_env="$1"

  if [[ -z "$conda_env" ]]; then
      echo -e "${FG_RED}[!] Error: No Conda environment name provided.${FG_RESET}"
      return 1
  fi

  echo -e "${FG_BLUE}[*] Building and activating Conda environment '$conda_env'...${FG_RESET}"
  . "${PACKAGE_DIR}/envs/conda/build_conda_env.sh" -c "$conda_env"

  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$conda_env"

  if [[ "$(conda info --envs | grep '*' | awk '{print $1}')" == "$conda_env" ]]; then
      echo -e "${FG_GREEN}[*] Conda environment '$conda_env' activated successfully.${FG_RESET}"
  else
      echo -e "${FG_RED}[!] Failed to activate the '$conda_env' environment.${FG_RESET}"
      exit 1
  fi
}

# Function to configure AWS CLI with environment variables from a .env file
configure_aws_from_env() {
    echo -e "${FG_BLUE}[*] Configuring AWS CLI from .env file...${FG_RESET}"

    if [[ -f .env ]]; then
        # Load each line, excluding comments and blank lines, and export key-value pairs
        while IFS= read -r line || [ -n "$line" ]; do
            # Ignore lines starting with # or without an equal sign
            if [[ "$line" =~ ^[[:space:]]*# ]] || [[ ! "$line" =~ = ]]; then
                continue
            fi
            export "$line"
        done < .env
        echo -e "${FG_GREEN}[*] .env file loaded successfully.${FG_RESET}"
    else
        echo -e "${FG_RED}[!] .env file not found!${FG_RESET}"
        return 1
    fi

    if [[ -z "$AWS_ACCESS_KEY_ID" || -z "$AWS_SECRET_ACCESS_KEY" || -z "$AWS_DEFAULT_REGION" ]]; then
        echo -e "${FG_RED}[!] Missing required AWS environment variables.${FG_RESET}"
        echo -e "${FG_YELLOW}Please ensure AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_DEFAULT_REGION are set in the .env file.${FG_RESET}"
        return 1
    fi

    aws configure set aws_access_key_id "${AWS_ACCESS_KEY_ID}"
    aws configure set aws_secret_access_key "${AWS_SECRET_ACCESS_KEY}"
    aws configure set region "${AWS_DEFAULT_REGION}"

    echo -e "${FG_GREEN}[*] AWS CLI configured successfully with .env values!${FG_RESET}"
}

load_env() {
  if [ -f .env ]; then
    while IFS= read -r line; do
      # Ignore comments and empty lines
      if [[ "$line" =~ ^# ]] || [[ -z "$line" ]]; then
        continue
      fi

      # Split key and value
      key=$(echo "$line" | cut -d= -f1)
      value=$(echo "$line" | cut -d= -f2-)

      # Remove surrounding quotes if they exist
      value=$(echo "$value" | sed 's/^"//' | sed 's/"$//')

      # Export the key-value pair
      export "$key=$value"
    done < .env
  else
    echo ".env file not found."
  fi
}

get_latest_model_version() {
  local bucket_name="$AWS_S3_BUCKET_NAME"
  local project_name="$PROJECT"

  if [ -z "$bucket_name" ] || [ -z "$project_name" ]; then
    echo "Error: AWS_S3_BUCKET_NAME or PROJECT is not set. Check your .env file."
    return 1
  fi

  local prefix="projects/${project_name}/models/"

  # List the model IDs and extract the latest version
  aws s3 ls "s3://${bucket_name}/${prefix}" --recursive | \
    awk -F "${prefix}" '{print $2}' | \
    awk -F'/' '{print $1}' | \
    sort -u | \
    tail -n 1
}

clean_directory() {
    local dir_path="$1"
    if [ -d "${dir_path}" ]; then
        rm -rf "${dir_path:?}"/*
        echo -e "${FG_YELLOW}[*] Cleaned directory: ${dir_path}.${FG_RESET}"
    fi
}

upload_to_s3() {
    local path="$1"           # First argument: Local path to upload
    local s3_path="$2"        # Second argument: S3 path for the upload
    shift 2                   # Remove the first two arguments (path and s3_path)
    local includes=("$@")     # Remaining arguments are inclusion patterns

    for include in "${includes[@]}"; do
        aws s3 cp "${path}" "${s3_path}" --recursive --exclude "*" --include "${include}" \
            --endpoint-url "${AWS_S3_ENDPOINT}" || {
            echo -e "${FG_RED}[!] Failed to upload ${include} to ${s3_path}.${FG_RESET}"
            exit 1
        }
    done
}
