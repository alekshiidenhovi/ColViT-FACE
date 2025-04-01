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
def finetune(dataset_dir: str):
    print("Starting finetuning process...")
    start_time = time.time()
    wandb_logger = init_wandb_logger()
    logger.info(f"Initializing finetuning experiment: {wandb_logger.experiment.name}")

    logger.info("Initializing training configs...")
    training_config = TrainingConfig(dataset_dir=dataset_dir)
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
