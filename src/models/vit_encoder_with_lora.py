import torch
from transformers import ViTModel
from models.vit_encoder import VitEncoder
from models.lora import LinearWithRSLoRA
from common.config import ModelConfig


class VitEncoderWithLoRA(torch.nn.Module):
    def __init__(self, vit_model: ViTModel, model_config: ModelConfig):
        super().__init__()
        self.encoder = VitEncoder(vit_model, model_config)
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

    def forward(self, input_tokens: torch.Tensor) -> torch.Tensor:
        return self.encoder(input_tokens)
