import os
import torch
import wandb
import wandb.wandb_run
import logging
import typing as T
from accelerate import Accelerator
from torch.utils.data import DataLoader
from common.metrics import recall_at_k
from models.utils import compute_similarity_scores
from pydantic import BaseModel, Field
from tqdm import tqdm


class ValidationMetrics(BaseModel):
    val_loss: float = Field(description="Validation loss", ge=0)
    val_recall_at_1: float = Field(description="Validation recall at 1", ge=0, le=1)
    val_recall_at_3: float = Field(description="Validation recall at 3", ge=0, le=1)
    val_recall_at_10: float = Field(description="Validation recall at 10", ge=0, le=1)


def validate(
    model: torch.nn.Module,
    accelerator: Accelerator,
    dataloader: DataLoader,
    wandb_run: wandb.wandb_run.Run,
    limit_batches: T.Optional[int] = None,
    is_test: bool = False,
) -> ValidationMetrics:
    """Run validation or test evaluation and return metrics."""
    model.eval()
    all_scores = []
    prefix = "test" if is_test else "val"
    desc = "Testing" if is_test else "Validating"

    progress_bar = tqdm(
        dataloader,
        desc=desc,
        leave=False,
        total=min(limit_batches, len(dataloader))
        if limit_batches is not None
        else None,
    )

    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            if limit_batches is not None and batch_idx >= limit_batches:
                break

            with accelerator.autocast():
                scores = compute_similarity_scores(batch, model)
                all_scores.append(scores)

    all_scores = torch.cat(all_scores, dim=0)
    all_targets = torch.zeros(
        all_scores.size(0), dtype=torch.int64, device=accelerator.device
    )

    loss = torch.nn.functional.cross_entropy(all_scores, all_targets)
    metrics = {f"{prefix}_loss": loss.item()}

    recall_values = [1, 3, 10]
    for recall_value in recall_values:
        recall = recall_at_k(all_scores, recall_value)
        metrics[f"{prefix}_recall_at_{recall_value}"] = recall

    wandb_run.log(metrics)
    model.train()

    return ValidationMetrics(**metrics)


def save_best_model(
    accelerator: Accelerator,
    epoch: int,
    logger: logging.Logger,
    model_checkpoint_path: str,
    enable_checkpointing: bool,
    val_metrics: ValidationMetrics,
    best_val_loss: float,
) -> float:
    """Save the best model and return the new best validation loss."""
    if enable_checkpointing and val_metrics.val_loss < best_val_loss:
        new_best_val_loss = val_metrics.val_loss
        checkpoint_path = os.path.join(
            model_checkpoint_path, f"best_model_epoch{epoch}.pt"
        )
        accelerator.save_state(checkpoint_path)
        logger.info(f"Saved new best model with val loss: {best_val_loss:.4f}")
        return new_best_val_loss
    return best_val_loss
