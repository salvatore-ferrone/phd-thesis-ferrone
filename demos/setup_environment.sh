#!/bin/bash
# filepath: /Users/sferrone/repos/phd-thesis-ferrone/demos/setup_environment.sh

set -e # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ENV_FILE="${SCRIPT_DIR}/environment.yaml"
ENV_NAME=$(grep "name:" "${ENV_FILE}" | head -n 1 | cut -d ":" -f 2 | tr -d " ")

echo "Setting up environment '${ENV_NAME}' from ${ENV_FILE}"

# Detect operating system and architecture
OS=$(uname -s)
ARCH=$(uname -m)

if [ "${OS}" == "Darwin" ]; then
    if [ "${ARCH}" == "arm64" ]; then
        echo "Detected macOS on Apple Silicon (M1/M2/M3)"
        echo "Creating native ARM64 conda environment..."
        CONDA_SUBDIR=osx-arm64 conda env create -f "${ENV_FILE}"
        
        # Set architecture-specific settings for this environment
        conda activate "${ENV_NAME}"
        conda config --env --set subdir osx-arm64
        conda deactivate
        
        echo "Environment created with native ARM64 packages"
    else
        echo "Detected macOS on Intel processor"
        conda env create -f "${ENV_FILE}"
    fi
else
    # Linux or other system
    echo "Detected ${OS} on ${ARCH}"
    conda env create -f "${ENV_FILE}"
fi

echo ""
echo "Environment '${ENV_NAME}' created successfully!"
echo "To activate: conda activate ${ENV_NAME}"
echo ""
echo "To rebuild tstrippy package after activation:"
echo "cd /path/to/tstrippy && ./build.sh"