from transformers import ViTConfig, ViTModel
from common.config import ModelConfig
import torch
import typing as T
import math


class ExtendedViTConfig(ViTConfig):
    def __init__(self, model_config: ModelConfig, **kwargs):
        super().__init__(**kwargs)
        self.reduced_dim: int = model_config.token_embedding_dim


class VitEncoder(ViTModel):
    """Vision Transformer (ViT) encoder with dimensionality reduction.

    This class wraps a pretrained ViT model and adds a linear projection layer to reduce the dimensionality of the output token embeddings

    Parameters
    ----------
    vit_model : ViTModel
        Pretrained Vision Transformer model to use as the base encoder
    model_config : ModelConfig
        Configuration object containing model architecture parameters

    Attributes
    ----------
    base_model : ViTModel
        The underlying pretrained ViT model
    dim_reduction : torch.nn.Linear
        Linear layer that reduces the token embedding dimension
    """

    def __init__(self, config: ExtendedViTConfig):
        super().__init__(config)
        hidden_dim = self.config.hidden_size

        self.dim_reduction = torch.nn.Linear(hidden_dim, config.reduced_dim)
        std_dev = 1 / math.sqrt(config.reduced_dim)
        self.dim_reduction.weight.data.normal_(0, std_dev)
        self.dim_reduction.bias.data.zero_()

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
        outputs = self(pixel_values).last_hidden_state
        output_tokens = self.dim_reduction(outputs)
        return output_tokens
