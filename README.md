# ColViT-FACE: Late Interaction for Face Identification

## Setup 

You need to have [uv](https://github.com/astral-sh/uv) installed to your system. [Install uv](https://github.com/astral-sh/uv) by running `curl -LsSf https://astral.sh/uv/install.sh | sh`.

Run the initialization bash script with the following command: `source scripts/activate-venv.sh`. This script will execute the following steps:
- Creates a virtual environment with Python 3.13 with uv (`scripts/create-venv.sh`)
- Activates the virtual environment (`scripts/activate-venv.sh`)
- Install packages to the virtual environment (`scripts/install-deps.sh`)

## W&B Logging
If you want to log your experiments with Weights & Biases, copy the `.env.template` file as `.env` file and fill in your W&B project name to `WANDB_PROJECT` variable.