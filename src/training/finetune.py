import click
import time
import torch
from tqdm import tqdm
from accelerate import Accelerator
from common.config import TrainingConfig
from common.nvidia import get_gpu_info_from_nvidia_smi
from common.logger import logger
from common.optimizer import get_adam8bit_optimizer
from common.wandb_logger import init_wandb_logger
from common.parameters import count_parameters
from common.metrics import recall_at_k
from datasets.casia_webface.dataloader import retrieve_dataloaders
from models.vit_encoder import VitEncoder, ExtendedViTConfig
from models.utils import compute_similarity_scores
from training.loops import validate, save_best_model
from transformers import (
    ViTImageProcessorFast,
    BitsAndBytesConfig,
    AutoConfig,
    ViTConfig,
)


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
@click.option(
    "--weight-decay",
    default=None,
    type=float,
    help="Weight decay for the optimizer",
)
@click.option(
    "--adam-beta1",
    default=None,
    type=float,
    help="Beta1 parameter for the Adam optimizer, used for the first moment estimate",
)
@click.option(
    "--adam-beta2",
    default=None,
    type=float,
    help="Beta2 parameter for the Adam optimizer, used for the second moment estimate",
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
    optimizer_config = training_config.get_optimizer_config()
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    logger.info("Initializing model and data modules...")
    vit_config: ViTConfig = AutoConfig.from_pretrained(model_config.pretrained_vit_name)
    extended_vit_config = ExtendedViTConfig(
        **vit_config.to_dict(), model_config=model_config
    )
    processor = ViTImageProcessorFast.from_pretrained(model_config.pretrained_vit_name)
    model = VitEncoder.from_pretrained(
        model_config.pretrained_vit_name,
        config=extended_vit_config,
        quantization_config=quantization_config,
    )
    print(model)
    train_dataloader, val_dataloader, test_dataloader = retrieve_dataloaders(
        processor, dataset_config
    )
    optimizer = get_adam8bit_optimizer(model, optimizer_config)

    gpu_info = get_gpu_info_from_nvidia_smi()

    logger.info("Logging training configs and GPU info to W&B...")
    total_params_count, trainable_params_count = count_parameters(model.encoder)
    wandb_logger.experiment.config.update(training_config.model_dump())
    wandb_logger.experiment.config.update({"gpu_info": gpu_info})
    wandb_logger.experiment.config.update(
        {"total_params": total_params_count, "trainable_params": trainable_params_count}
    )
    wandb_logger.watch(model)

    if finetuning_config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    accelerator = Accelerator(
        mixed_precision=finetuning_config.precision.value,
        gradient_accumulation_steps=finetuning_config.accumulate_grad_batches,
    )
    model, optimizer, train_dataloader, val_dataloader, test_dataloader = (
        accelerator.prepare(
            model, optimizer, train_dataloader, val_dataloader, test_dataloader
        )
    )

    global_step = 0
    best_val_loss = 0.0
    model.train()

    logger.info("Starting model finetuning...")
    for epoch in range(finetuning_config.max_epochs):
        logger.info(f"Starting epoch {epoch + 1} of {finetuning_config.max_epochs}")
        total_train_loss = 0

        train_progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch + 1}/{finetuning_config.max_epochs}",
            leave=True,
        )

        for batch in train_progress_bar:
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    scores = compute_similarity_scores(batch, model)
                    targets = torch.zeros(
                        scores.size(0), dtype=torch.int64, device=accelerator.device
                    )
                    loss = torch.nn.functional.cross_entropy(scores, targets)
                    total_train_loss += loss.item()
                    accelerator.backward(loss)
                    optimizer.step()
                    optimizer.zero_grad()

            wandb_logger.log_metrics({"train_loss": loss.item(), "epoch": epoch})
            recall_values = [1, 3, 10]
            for recall_value in recall_values:
                recall = recall_at_k(scores, recall_value)
                wandb_logger.log_metrics(
                    {f"train_recall_at_{recall_value}": recall, "epoch": epoch}
                )

            global_step += 1
            train_progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            if global_step % finetuning_config.val_check_interval == 0:
                train_progress_bar.set_postfix({"status": "Running validation..."})
                val_metrics = validate(
                    model=model,
                    accelerator=accelerator,
                    dataloader=val_dataloader,
                    wandb_logger=wandb_logger,
                    limit_batches=finetuning_config.val_check_interval,
                    is_test=False,
                )
                save_best_model(
                    accelerator=accelerator,
                    epoch=epoch,
                    logger=logger,
                    model_checkpoint_path=finetuning_config.model_checkpoint_path,
                    enable_checkpointing=finetuning_config.enable_checkpointing,
                    val_metrics=val_metrics,
                    best_val_loss=best_val_loss,
                )
                train_progress_bar.set_postfix(
                    {
                        "train_loss": f"{loss.item():.4f}",
                        "val_loss": f"{val_metrics.val_loss:.4f}",
                    }
                )

        wandb_logger.log_metrics(
            {
                "train_epoch_loss": total_train_loss / len(train_dataloader),
                "epoch": epoch,
            }
        )
        logger.info(f"Running validation for epoch {epoch + 1}")
        val_metrics = validate(
            model,
            val_dataloader,
            accelerator,
            wandb_logger,
            limit_batches=finetuning_config.limit_val_batches,
            is_test=False,
        )

    logger.info("Running final evaluation on test set...")
    test_metrics = validate(
        model, test_dataloader, accelerator, wandb_logger, is_test=True
    )
    logger.info(f"Test metrics: {test_metrics}")

    end_time = time.time()
    logger.info(f"Finished finetuning in {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    finetune()
