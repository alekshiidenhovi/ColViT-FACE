import click
import time

from models.colvit import ColViT
from common.config import TrainingConfig
from common.nvidia import get_gpu_info_from_nvidia_smi
from common.logger import logger
from common.wandb_logger import init_wandb_logger
from common.parameters import count_parameters
from datasets.casia_webface.data_module import CASIAFaceDataModule
from lightning.pytorch import Trainer, seed_everything


@click.command()
@click.option(
    "--dataset-dir", help="Absolute path to the CASIA dataset directory", type=str
)
@click.option(
    "--train-batch-size", default=None, type=int, help="Batch size for training"
)
@click.option(
    "--val-batch-size", default=None, type=int, help="Batch size for validation"
)
@click.option(
    "--test-batch-size", default=None, type=int, help="Batch size for testing"
)
@click.option(
    "--num-workers", default=None, type=int, help="Number of workers for data loading"
)
@click.option("--img-size", default=None, type=int, help="Size of the input images")
@click.option(
    "--train-val-test-split",
    default=None,
    type=(float, float, float),
    help="Train, validation, and test split proportions",
)
@click.option(
    "--train-num-negative-samples",
    default=None,
    type=int,
    help="Number of negative samples for training",
)
@click.option(
    "--val-num-negative-samples",
    default=None,
    type=int,
    help="Number of negative samples for validation",
)
@click.option(
    "--test-num-negative-samples",
    default=None,
    type=int,
    help="Number of negative samples for testing",
)
@click.option(
    "--pretrained-vit-name",
    default=None,
    type=str,
    help="Name of the pretrained vision transformer to use",
)
@click.option(
    "--token-embedding-dim",
    default=None,
    type=int,
    help="Final dimension of the token embeddings",
)
@click.option(
    "--learning-rate", default=None, type=float, help="Learning rate of the model"
)
@click.option(
    "--lora-rank", default=None, type=int, help="Rank of the LoRA decomposition matrix"
)
@click.option(
    "--lora-alpha",
    default=None,
    type=int,
    help="Scaling factor for the LoRA decomposition matrix",
)
@click.option(
    "--accelerator",
    default=None,
    type=str,
    help="Compute accelerator to use for training",
)
@click.option(
    "--warmup-steps",
    default=None,
    type=int,
    help="Number of steps to warmup the learning rate",
)
@click.option(
    "--enable-checkpointing",
    default=None,
    type=bool,
    help="Enables checkpointing of the latest training epoch",
)
@click.option(
    "--val-check-interval",
    default=None,
    type=int,
    help="The interval of training batches to run validation",
)
@click.option(
    "--limit-val-batches",
    default=None,
    type=int,
    help="Number of validation batches to run",
)
@click.option(
    "--precision",
    default=None,
    type=str,
    help="Precision of gradients and the model used during finetuning",
)
@click.option(
    "--model-checkpoint-path",
    default=None,
    type=str,
    help="Folder for saving model checkpoints during training",
)
@click.option(
    "--max-epochs", default=None, type=int, help="Maximum number of epochs to train for"
)
@click.option(
    "--devices", default=None, type=str, help="GPU devices to use for training"
)
@click.option(
    "--accumulate-grad-batches",
    default=None,
    type=int,
    help="Number of batches to accumulate gradients before updating the model",
)
@click.option(
    "--seed", default=None, type=int, help="Seed for training reproducibility"
)
def finetune(**kwargs):
    print("Starting finetuning process...")
    start_time = time.time()
    wandb_logger = init_wandb_logger()
    logger.info(f"Initializing finetuning experiment: {wandb_logger.experiment.name}")

    logger.info("Initializing training configs...")
    valid_fields = set(TrainingConfig.model_fields.keys())
    config_kwargs = {
        k: v for k, v in kwargs.items() if v is not None and k in valid_fields
    }
    training_config = TrainingConfig(**config_kwargs)
    model_config = training_config.get_model_config()
    dataset_config = training_config.get_dataset_config()
    finetuning_config = training_config.get_finetuning_config()

    seed_everything(training_config.seed)

    logger.info("Initializing model and data modules...")
    model = ColViT(model_config)
    datamodule = CASIAFaceDataModule(dataset_config)
    gpu_info = get_gpu_info_from_nvidia_smi()

    logger.info("Logging training configs and GPU info to W&B...")
    total_params_count, trainable_params_count = count_parameters(model.encoder)
    wandb_logger.experiment.config.update(training_config.model_dump())
    wandb_logger.experiment.config.update({"gpu_info": gpu_info})
    wandb_logger.experiment.config.update(
        {"total_params": total_params_count, "trainable_params": trainable_params_count}
    )
    wandb_logger.watch(model)

    logger.info("Initializing Lightning Trainer...")
    trainer = Trainer(
        logger=wandb_logger,
        devices=finetuning_config.devices,
        accelerator=finetuning_config.accelerator,
        precision=finetuning_config.precision,
        val_check_interval=finetuning_config.val_check_interval,
        limit_val_batches=finetuning_config.limit_val_batches,
        enable_checkpointing=finetuning_config.enable_checkpointing,
        default_root_dir=finetuning_config.model_checkpoint_path,
        max_epochs=finetuning_config.max_epochs,
        accumulate_grad_batches=finetuning_config.accumulate_grad_batches,
    )

    logger.info("Starting model finetuning...")
    trainer.fit(model, datamodule=datamodule)
    end_time = time.time()
    logger.info(f"Finished finetuning in {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    finetune()
