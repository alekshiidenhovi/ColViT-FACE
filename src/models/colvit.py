import torch
import pytorch_lightning as pl
from src.models.vit_encoder import VitEncoder
from src.config import TrainingConfig
from src.models.utils import compute_metrics
from src.models.lora import LinearWithLoRA


class ColViT(pl.LightningModule):
    def __init__(self, config: TrainingConfig):
        self.encoder = VitEncoder(
            config.token_embedding_dim, config.pretrained_vit_name
        )
        self.config = config

        for param in self.encoder.parameters():
            param.requires_grad = False

        self._add_lora_layers()

    def _add_lora_layers(self):
        for block in self.encoder.model.blocks:
            block.attn.qkv = LinearWithLoRA(
                block.attn.qkv, self.config.rank, self.config.alpha
            )
            block.attn.proj = LinearWithLoRA(
                block.attn.proj, self.config.rank, self.config.alpha
            )
            block.mlp.fc1 = LinearWithLoRA(
                block.mlp.fc1, self.config.rank, self.config.alpha
            )
            block.mlp.fc2 = LinearWithLoRA(
                block.mlp.fc2, self.config.rank, self.config.alpha
            )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.encoder(tokens)

    def training_step(self, batch: torch.Tensor, batch_idx):
        loss, recall_at_1, recall_at_3 = compute_metrics(batch, self.encoder)
        self.log("train_loss", loss)
        self.log("train_recall_at_1", recall_at_1)
        self.log("train_recall_at_3", recall_at_3)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx):
        loss, recall_at_1, recall_at_3 = compute_metrics(batch, self.encoder)
        self.log("val_loss", loss)
        self.log("val_recall_at_1", recall_at_1)
        self.log("val_recall_at_3", recall_at_3)
        return loss

    def testing_step(self, batch: torch.Tensor, batch_idx):
        loss, recall_at_1, recall_at_3 = compute_metrics(batch, self.encoder)
        self.log("test_loss", loss)
        self.log("test_recall_at_1", recall_at_1)
        self.log("test_recall_at_3", recall_at_3)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)
        return optimizer
