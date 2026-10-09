#!/bin/bash
# Source this (`source bash/load_env.sh`); running it directly activates only a subshell.

PROJ_ENV=${1:-"syngeny-env"}

# 1. Check if PROJ_ENV is a valid Conda environment
if ! conda env list | grep -q "\b$PROJ_ENV\b"; then
    echo "Error: Conda environment '$PROJ_ENV' does not exist." 
    echo "Run ./bash/setup.sh <PROJ_DIR> <PROJ_ENV> first."
    return 1 2>/dev/null || exit 1
fi
# Load conda's shell functions so `conda activate` works in non-interactive shells
eval "$(conda shell.bash hook)"
conda activate $PROJ_ENV

# 2. Check if PROJ_DIR is set and exists
if [[ -z "$PROJ_DIR" || ! -e "$PROJ_DIR" ]]; then
    echo "Error: environment variable $PROJ_DIR is not set."
    echo "Set PROJ_DIR environment variable to the project directory."
    return 1 2>/dev/null || exit 1
fi
cd $PROJ_DIR