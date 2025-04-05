from transformers import ViTConfig, ViTModel
from common.config import ModelConfig
from common.logger import logger
from safetensors.torch import load_file
import torch
import os
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
        super().__init__(config, add_pooling_layer=False)
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
        embedding_output = self.embeddings(
            pixel_values=pixel_values,
            bool_masked_pos=bool_masked_pos,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = encoder_outputs[0]
        normalized_last_hidden_state = self.layernorm(last_hidden_state)
        output_tokens = self.dim_reduction(normalized_last_hidden_state)
        return output_tokens
    
    def load_from_checkpoint(self, checkpoint_dir_path: str):
        """Load the model from a checkpoint directory.
        
        This function loads the model from a checkpoint directory and updates the model's state dictionary.
        
        Parameters
        ----------
        checkpoint_dir_path : str
            The path to the checkpoint directory
        """
        checkpoint_path = os.path.join(checkpoint_dir_path, "model.safetensors")
        state_dict = load_file(checkpoint_path)
    
    missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
    
    if len(missing_keys):
        if 'dim_reduction.weight' in missing_keys or 'dim_reduction.bias' in missing_keys:
            logger.warning("Missing dim_reduction layer keys. This might be expected if loading a base model.")
        else:
            logger.warning(f"Missing keys: {missing_keys}")
    
    if len(unexpected_keys):
        # Filter out quantization-related keys
        non_quant_unexpected_keys = [
            key 
            for key in unexpected_keys 
            if not any(
                suffix in key 
                for suffix in ['.absmax', '.quant_state', '.quant_map', '.nested_absmax', '.nested_quant_map']
            )
        ]
        
        if non_quant_unexpected_keys:
            logger.warning(f"Unexpected non-quantization keys: {non_quant_unexpected_keys}")
        else:
            logger.info(f"All unexpected keys are related to quantization and can be safely ignored.")
        
        logger.info(f"Loaded model successfully from checkpoint path: {checkpoint_path}")
        
        
