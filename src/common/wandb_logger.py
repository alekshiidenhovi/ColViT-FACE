import wandb
import os
from dotenv import load_dotenv
from pytorch_lightning.loggers import WandbLogger

def init_wandb_logger():
    """
    Initialize and configure a Weights & Biases logger.

    This function loads environment variables, authenticates with W&B using the API key,
    and creates a WandbLogger instance for experiment tracking.

    Returns
    -------
    WandbLogger
        Configured W&B logger instance that can be used with PyTorch Lightning.

    Notes
    -----
    Requires the following environment variables to be set:
    - WANDB_API_KEY: The Weights & Biases API key for authentication
    - WANDB_PROJECT: The name of the W&B project to log to
    """
    load_dotenv()
    wandb_api_key = os.getenv("WANDB_API_KEY")
    wandb_project = os.getenv("WANDB_PROJECT")
    wandb.login(key=wandb_api_key)
    logger = WandbLogger(project=wandb_project)
    return logger

