#!/bin/bash
PROJECT_DIR=${1:-$(pwd)}

source "$PROJECT_DIR/scripts/activate-venv.sh"
uv pip compile requirements.in requirements-dev.in --output-file requirements-dev.txt