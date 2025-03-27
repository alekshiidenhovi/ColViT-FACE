import timm
import torch.nn as nn
import torch
from einops import einsum


class VitEncoder(nn.Module):
    def __init__(
        self,
        reduced_dim: int,
        model_name: str = "vit_small_patch16_384.augreg_in21k_ft_in1k",
    ):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        hidden_dim: int | None = getattr(self.model, "embed_dim", None)
        if hidden_dim is None:
            raise ValueError("Cannot find embed_dim attribute on model.")
        self.weights = nn.Parameter(torch.randn(hidden_dim, reduced_dim))
        self.bias = nn.Parameter(torch.zeros(reduced_dim))

    def forward(self, x):
        tokens = self.model.forward_features(x)
        tokens = (
            einsum(
                tokens,
                self.weights,
                "batch_size seq_len hidden_dim, hidden_dim reduced_dim -> batch_size seq_len reduced_dim",
            )
            + self.bias
        )
        return tokens
