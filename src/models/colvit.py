import torch
import torch.nn.functional as F
import lightning as L
from models.vit_encoder import VitEncoder
from models.utils import compute_similarity_scores
from models.lora import LinearWithRSLoRA
from common.config import ModelConfig, OptimizerConfig
from common.metrics import recall_at_k
from common.optimizer import get_adam8bit_optimizer


class ColViT(L.LightningModule):
    def __init__(self, model_config: ModelConfig, optimizer_config: OptimizerConfig):
        super().__init__()
        self.encoder = VitEncoder(
            model_config.token_embedding_dim, model_config.pretrained_vit_name
        )
        self.optimizer_config = optimizer_config

        for param in self.encoder.parameters():
            param.requires_grad = False

        for block in self.encoder.model.blocks:
            block.attn.qkv = LinearWithRSLoRA(
                block.attn.qkv,
                model_config.lora_rank,
                model_config.lora_alpha,
            )
            block.attn.proj = LinearWithRSLoRA(
                block.attn.proj,
                model_config.lora_rank,
                model_config.lora_alpha,
            )
            block.mlp.fc1 = LinearWithRSLoRA(
                block.mlp.fc1, model_config.lora_rank, model_config.lora_alpha
            )
            block.mlp.fc2 = LinearWithRSLoRA(
                block.mlp.fc2, model_config.lora_rank, model_config.lora_alpha
            )

        for param in self.encoder.dim_reduction.parameters():
            param.requires_grad = True

        for name, param in self.encoder.named_parameters():
            if "lora" in name:
                param.requires_grad = True

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.encoder(tokens)

    def training_step(self, batch: torch.Tensor, batch_idx):
        scores = compute_similarity_scores(batch, self.encoder)
        targets = torch.zeros(scores.size(0), dtype=torch.int64, device=self.device)
        loss = F.cross_entropy(scores, targets)
        recall_at_1 = recall_at_k(scores, 1)
        self.log("train_loss", loss)
        self.log("train_recall_at_1", recall_at_1)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx):
        scores = compute_similarity_scores(batch, self.encoder)
        targets = torch.zeros(scores.size(0), dtype=torch.int64, device=self.device)
        loss = F.cross_entropy(scores, targets)
        recall_at_1 = recall_at_k(scores, 1)
        recall_at_3 = recall_at_k(scores, 3)
        recall_at_10 = recall_at_k(scores, 10)
        self.log("val_loss", loss)
        self.log("val_recall_at_1", recall_at_1)
        self.log("val_recall_at_3", recall_at_3)
        self.log("val_recall_at_10", recall_at_10)
        return loss

    def testing_step(self, batch: torch.Tensor, batch_idx):
        scores = compute_similarity_scores(batch, self.encoder)
        targets = torch.zeros(scores.size(0), dtype=torch.int64, device=self.device)
        loss = F.cross_entropy(scores, targets)
        recall_at_1 = recall_at_k(scores, 1)
        recall_at_3 = recall_at_k(scores, 3)
        recall_at_10 = recall_at_k(scores, 10)
        recall_at_100 = recall_at_k(scores, 100)
        self.log("test_loss", loss)
        self.log("test_recall_at_1", recall_at_1)
        self.log("test_recall_at_3", recall_at_3)
        self.log("test_recall_at_10", recall_at_10)
        self.log("test_recall_at_100", recall_at_100)
        return loss

    def configure_optimizers(self):
        optimizer = get_adam8bit_optimizer(self.encoder, self.optimizer_config)
        return optimizer
