# ColViT-FACE: Late Interaction for Face Identification

## Setup 

Run the initialization bash script with the following command: `source scripts/activate-venv.sh`. This script will execute the following steps:
- Installs [uv](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer) as a package manager.
- Creates a virtual environment with Python3.11 with uv (`scripts/create-venv.sh`)
- Activates the virtual environment (`scripts/activate-venv.sh`)
- Install packages to the virtual environment (`scripts/install-deps.sh`)

## W&B Logging
If you want to log your experiments with Weights & Biases, copy the `.env.template` file as `.env` file and fill in your W&B project name to `WANDB_PROJECT` variable.