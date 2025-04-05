import wandb
import os
from dotenv import load_dotenv



def init_wandb_api_client():
    """Initialize a W&B API client."""
    load_dotenv()
    WANDB_API_KEY = os.getenv("WANDB_API_KEY")
    return wandb.Api(api_key=WANDB_API_KEY)


def init_wandb_run():
    """
    Initialize and configure a Weights & Biases logger.

    This function loads environment variables from a .env file, authenticates with W&B using the API key,
    and initializes a new W&B run for experiment tracking. The run is configured with the project and entity
    specified in the environment variables.

    Returns
    -------
    wandb.wandb_run.Run
        An initialized W&B run object that can be used to log metrics, artifacts and other experiment data.
        The run will be associated with the specified project and entity.

    Notes
    -----
    Required environment variables in .env file:
    - WANDB_API_KEY: The Weights & Biases API key for authentication
    - WANDB_PROJECT: The name of the W&B project to log experiments to
    - WANDB_ENTITY: The W&B username or team name that owns the project
    """
    load_dotenv()
    WANDB_API_KEY = os.getenv("WANDB_API_KEY")
    WANDB_PROJECT = os.getenv("WANDB_PROJECT")
    WANDB_ENTITY = os.getenv("WANDB_ENTITY")

    wandb.login(key=WANDB_API_KEY)

    run = wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY)
    return run
