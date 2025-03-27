#!/bin/bash
source ~/ColViT-FACE/scripts/activate-env.sh
uv pip compile requirements.in requirements-dev.in --output-file requirements-dev.txt