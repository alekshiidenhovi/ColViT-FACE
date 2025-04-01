import wandb
import os
from dotenv import load_dotenv
from lightning.pytorch.loggers import WandbLogger


def init_wandb_logger():
    """
    Initialize and configure a Weights & Biases logger.

    This function loads environment variables, authenticates with W&B using the API key,
    and creates a WandbLogger instance for experiment tracking.

    Returns
    -------
    WandbLogger
        The configured PyTorch Lightning W&B logger instance

    Notes
    -----
    Requires the following environment variables to be set:
    - WANDB_API_KEY: The Weights & Biases API key for authentication
    - WANDB_PROJECT: The name of the W&B project to log to
    - WANDB_ENTITY: The W&B username or team name
    """
    load_dotenv()
    WANDB_API_KEY = os.getenv("WANDB_API_KEY")
    WANDB_PROJECT = os.getenv("WANDB_PROJECT")
    WANDB_ENTITY = os.getenv("WANDB_ENTITY")

    wandb.login(key=WANDB_API_KEY)

    experiment = wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY)
    logger = WandbLogger(project=WANDB_PROJECT, experiment=experiment)
    return logger
