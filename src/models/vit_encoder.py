import timm
import torch.nn as nn
import torch


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
        
        self.dim_reduction = nn.Linear(hidden_dim, reduced_dim)
        std_dev = 1 / torch.sqrt(torch.tensor(reduced_dim).float())
        self.dim_reduction.weight.data.normal_(0, std_dev)
        self.dim_reduction.bias.data.zero_()

    def forward(self, input_tokens: torch.Tensor):
        intermediate_tokens = self.model.forward_features(input_tokens)
        output_tokens = self.dim_reduction(intermediate_tokens)
        return output_tokens
