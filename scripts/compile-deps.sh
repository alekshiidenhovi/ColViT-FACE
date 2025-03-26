#!/bin/bash
source ~/ColViT-FACE/scripts/activate-env.sh
uv pip compile requirements.in --output-file requirements.txt
uv pip compile requirements-dev.in --output-file requirements-dev.txt