#!/bin/bash
PROJECT_DIR=${1:-$(pwd)}

source "$PROJECT_DIR/scripts/install-uv.sh"
source "$PROJECT_DIR/scripts/create-venv.sh"
source "$PROJECT_DIR/scripts/sync-deps-dev.sh"