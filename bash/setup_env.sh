#!/bin/bash

PROJ_DIR=${1:-"$HOME/research/projects/syngeny"}
PROJ_ENV=${2:-"syngeny-env"}

# Make `conda activate` available inside this non-interactive shell
source "$(conda info --base)/etc/profile.d/conda.sh"

# check if PROJ_DIR exists
ls $PROJ_DIR
if [ ! -d "${PROJ_DIR}" ]; then
    echo $(pwd)
    echo "Error: Directory $PROJ_DIR does not exist. Create project directory first."
    exit 1
fi

if conda env list | grep -q "\b$PROJ_ENV\b"; then
    echo "Environment already exists!"
    read -p "Do you want to continue? (y/n): " choice
    case "$choice" in 
    ( y|Y|[yY][eE][sS] )
        echo "Continuing without creating a new environment.." 
        ;;
    ( * )
        exit 1
        ;;
    esac
else 
    echo "Creating new environment $PROJ_ENV"
    conda env create -f environment.yml -n $PROJ_ENV
fi

# Activate project env and set project directory

conda activate $PROJ_ENV
echo "Activated Conda environment: $PROJ_ENV"
echo "Installing project in editable mode"
pip install -e .
echo "Setting environment variable PROJ_DIR to $PROJ_DIR"
conda env config vars set PROJ_DIR=$PROJ_DIR
cd $PROJ_DIR