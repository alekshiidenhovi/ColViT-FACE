from transformers import ViTModel
from common.config import ModelConfig
import torch
import math


class VitEncoder(torch.nn.Module):
    def __init__(
        self,
        vit_model: ViTModel,
        model_config: ModelConfig,
    ):
        super().__init__()
        self.base_model = vit_model
        hidden_dim = self.base_model.config.hidden_size

        self.dim_reduction = torch.nn.Linear(hidden_dim, model_config.token_embedding_dim)
        std_dev = 1 / math.sqrt(model_config.token_embedding_dim)
        self.dim_reduction.weight.data.normal_(0, std_dev)
        self.dim_reduction.bias.data.zero_()

    def forward(self, input_tokens: torch.Tensor):
        outputs = self.base_model(input_tokens).last_hidden_state
        output_tokens = self.dim_reduction(outputs)
        return output_tokens
