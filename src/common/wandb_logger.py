import wandb
import os
from dotenv import load_dotenv
from pytorch_lightning.loggers import WandbLogger

def init_wandb_logger():
    load_dotenv()
    wandb_api_key = os.getenv("WANDB_API_KEY")
    wandb_project = os.getenv("WANDB_PROJECT")
    wandb.login(key=wandb_api_key)
    logger = WandbLogger(project=wandb_project)
    return logger

