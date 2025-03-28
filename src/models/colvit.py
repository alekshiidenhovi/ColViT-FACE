import torch
from torch import Tensor
import pytorch_lightning as pl
from src.models.vit_encoder import VitEncoder
from src.config import TrainingConfig
from src.training.scoring import maxsim


class ColViT(pl.LightningModule):
    def __init__(self, config: TrainingConfig):
        self.encoder = VitEncoder(
            config.token_embedding_dim, config.pretrained_vit_name
        )

    def forward(self, tokens: Tensor) -> Tensor:
        return self.encoder(tokens)

    def training_step(self, batch, batch_idx):
        query_image, positive_image, negative_images, _, _, _ = batch

        query_repr = self(query_image)  # [batch, 1, seq_len, token_embedding_dim]
        all_gallery = torch.cat(
            [positive_image, negative_images], dim=1
        )  # [batch, 1+num_neg, H, W]
        gallery_repr = self(
            all_gallery
        )  # [batch, 1+num_neg, seq_len, token_embedding_dim]

        scores = maxsim(query_repr, gallery_repr)  # [batch, 1+num_neg]

        targets = torch.zeros_like(scores)
        targets[:, 0] = 1

        loss = torch.nn.functional.cross_entropy(scores, targets)
        return loss

    def validation_step(self, batch, batch_idx):
        query_image, positive_image, negative_images, _, _, _ = batch

        query_repr = self(query_image)  # [batch, 1, seq_len, token_embedding_dim]
        all_gallery = torch.cat(
            [positive_image, negative_images], dim=1
        )  # [batch, 1+num_neg, H, W]
        gallery_repr = self(
            all_gallery
        )  # [batch, 1+num_neg, seq_len, token_embedding_dim]

        scores = maxsim(query_repr, gallery_repr)  # [batch, 1+num_neg]

        targets = torch.zeros_like(scores)
        targets[:, 0] = 1

        loss = torch.nn.functional.cross_entropy(scores, targets)
        return loss

    def testing_step(self, batch, batch_idx):
        query_image, positive_image, negative_images, _, _, _ = batch

        query_repr = self(query_image)  # [batch, 1, seq_len, token_embedding_dim]
        all_gallery = torch.cat(
            [positive_image, negative_images], dim=1
        )  # [batch, 1+num_neg, H, W]
        gallery_repr = self(
            all_gallery
        )  # [batch, 1+num_neg, seq_len, token_embedding_dim]

        scores = maxsim(query_repr, gallery_repr)  # [batch, 1+num_neg]

        targets = torch.zeros_like(scores)
        targets[:, 0] = 1

        loss = torch.nn.functional.cross_entropy(scores, targets)
        return loss
