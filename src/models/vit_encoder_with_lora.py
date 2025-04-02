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

        for block in self.encoder.base_model.encoder.layer:
            block.attention.attention.query = LinearWithRSLoRA(
                block.attention.attention.query,
                rank=model_config.lora_rank,
                alpha=model_config.lora_alpha,
            )
            block.attention.attention.key = LinearWithRSLoRA(
                block.attention.attention.key,
                rank=model_config.lora_rank,
                alpha=model_config.lora_alpha,
            )
            block.attention.attention.value = LinearWithRSLoRA(
                block.attention.attention.value,
                rank=model_config.lora_rank,
                alpha=model_config.lora_alpha,
            )
            block.attention.output.dense = LinearWithRSLoRA(
                block.attention.output.dense,
                rank=model_config.lora_rank,
                alpha=model_config.lora_alpha,
            )
            block.intermediate.dense = LinearWithRSLoRA(
                block.intermediate.dense,
                rank=model_config.lora_rank,
                alpha=model_config.lora_alpha,
            )
            block.output.dense = LinearWithRSLoRA(
                block.output.dense,
                rank=model_config.lora_rank,
                alpha=model_config.lora_alpha,
            )

        for param in self.encoder.dim_reduction.parameters():
            param.requires_grad = True

        for name, param in self.encoder.named_parameters():
            if "lora" in name:
                param.requires_grad = True

    def forward(self, input_tokens: torch.Tensor) -> torch.Tensor:
        return self.encoder(input_tokens)
