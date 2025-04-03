import typing as T
import pandas as pd
from transformers import PreTrainedModel
from pydantic import BaseModel, Field


class ParameterInfo(BaseModel):
    total_params: int = Field(description="Total number of parameters in the model")
    trainable_params: int = Field(description="Number of trainable parameters in the model")
    trainable_percent: float = Field(description="Percentage of trainable parameters")
    trainable_layers: list[str] = Field(description="List of layer names with trainable parameters")
    param_counts_by_layer: pd.DataFrame = Field(description="DataFrame with parameter counts by layer")


def collect_parameter_info(model: PreTrainedModel) -> T.Dict[str, T.Any]:
    """Collects information about the model non-trainable and trainable parameters.

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model to analyze

    Returns
    -------
    dict
        Dictionary containing model parameter statistics:
        - total_params: Total number of parameters
        - trainable_params: Number of trainable parameters
        - trainable_percent: Percentage of trainable parameters
        - param_size_mb: Approximate model size in MB
        - trainable_layers: List of layer names with trainable parameters
        - param_counts_by_layer: DataFrame with parameter counts by layer
    """
    total_param_count = 0
    trainable_param_count = 0
    trainable_layers = set()
    layer_stats = []

    for name, param in model.named_parameters():
        param_count = param.numel()
        total_param_count += param_count
        is_trainable = param.requires_grad
        
        if is_trainable:
            trainable_param_count += param_count
            top_level_name = name.split('.')[0]
            trainable_layers.add(top_level_name)
            
        layer_stats.append({
            'name': name,
            'shape': list(param.shape),
            'params': param_count,
            'trainable': is_trainable,
            'dtype': str(param.dtype).split('.')[-1]
        })
    
    trainable_percent = (trainable_param_count / total_param_count * 100) if total_param_count > 0 else 0
    param_counts_df = pd.DataFrame(layer_stats)
    
    return {
        'total_params': total_param_count,
        'trainable_params': trainable_param_count,
        'trainable_percent': trainable_percent,
        'trainable_layers': list(trainable_layers),
        'param_counts_by_layer': param_counts_df
    }