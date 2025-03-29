#!/bin/bash
PROJECT_DIR=${1:-$(pwd)}

source "$PROJECT_DIR/scripts/activate-venv.sh"
uv pip sync requirements-dev.txt