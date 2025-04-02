import torch
import typing as T
from transformers import ViTConfig, ViTPreTrainedModel, BitsAndBytesConfig
from models.vit_encoder import VitEncoder
from models.lora import LinearWithRSLoRA
from common.config import ModelConfig


class VitEncoderWithLoRA(ViTPreTrainedModel):
    def __init__(
        self,
        vit_config: ViTConfig,
        model_config: ModelConfig,
        quantization_config: T.Optional[BitsAndBytesConfig] = None,
    ):
        super().__init__(vit_config)
        self.encoder = VitEncoder(
            vit_config=vit_config,
            model_config=model_config,
            quantization_config=quantization_config,
        )
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

    def forward(
        self,
        pixel_values: T.Optional[torch.Tensor] = None,
        bool_masked_pos: T.Optional[torch.BoolTensor] = None,
        head_mask: T.Optional[torch.Tensor] = None,
        output_attentions: T.Optional[bool] = None,
        output_hidden_states: T.Optional[bool] = None,
        interpolate_pos_encoding: T.Optional[bool] = None,
        return_dict: T.Optional[bool] = None,
    ) -> torch.Tensor:
        """Forward pass of the model.

        Parameters
        ----------
        pixel_values : torch.Tensor, optional
            Pixel values of the input images
        bool_masked_pos : torch.BoolTensor, optional
            Boolean mask for masked modeling
        head_mask : torch.Tensor, optional
            Mask to nullify selected heads of the attention layers
        output_attentions : bool, optional
            Whether to return attention weights
        output_hidden_states : bool, optional
            Whether to return hidden states
        """
        return self.encoder(pixel_values)
