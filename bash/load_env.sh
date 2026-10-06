#!/bin/bash

PROJ_ENV=${1:-"syngeny-env"}

# 1. Check if PROJ_ENV is a valid Conda environment
if ! conda env list | grep -q "\b$PROJ_ENV\b"; then
    echo "Error: Conda environment '$PROJ_ENV' does not exist." 
    echo "Run ./bash/setup.sh <PROJ_DIR> <PROJ_ENV> first."
    exit 1
fi
conda activate $PROJ_ENV

# 2. Check if PROJ_DIR is set and exists
if [[ -z "$PROJ_DIR" || ! -e "$PROJ_DIR" ]]; then
    echo "Error: environment variable $PROJ_DIR is not set."
    echo "Set PROJ_DIR environment variable to the project directory."
    exit 1
fi
cd $PROJ_DIR