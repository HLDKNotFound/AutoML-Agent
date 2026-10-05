#!/usr/bin/env bash
set -e

# Activate Conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ml-env

export PYTHONPATH=.
echo "Starting ML Automation Agent FastAPI Backend on http://127.0.0.1:8000 ..."
python -m backend.app.main
