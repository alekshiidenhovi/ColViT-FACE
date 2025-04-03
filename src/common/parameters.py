import typing as T
import pandas as pd
from transformers import PreTrainedModel
from common.logger import logger
from peft import PeftModel
from pydantic import BaseModel, Field


class ParameterInfo(BaseModel):
    total_params: int = Field(description="Total number of parameters in the model")
    trainable_params: int = Field(description="Number of trainable parameters in the model")
    trainable_percent: float = Field(description="Percentage of trainable parameters")
    trainable_layers: list[str] = Field(description="List of layer names with trainable parameters")
    param_counts_by_layer: pd.DataFrame = Field(description="DataFrame with parameter counts by layer")
    
    class Config:
        arbitrary_types_allowed = True


def collect_parameter_info(model: T.Union[PreTrainedModel, "PeftModel"]) -> ParameterInfo:
    """Collects information about the model non-trainable and trainable parameters.

    Parameters
    ----------
    model : Union[PreTrainedModel, PeftModel]
        PyTorch model to analyze (can be a regular model or a PEFT-wrapped model)

    Returns
    -------
    ParameterInfo
        Object containing model parameter statistics with detailed hierarchy of trainable layers
    """
    
    total_param_count = 0
    trainable_param_count = 0
    trainable_layers = []
    layer_stats = []
    trainable_param_names = set()
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_param_count += param_count
        is_trainable = param.requires_grad
        
        if is_trainable:
            trainable_param_count += param_count
            trainable_param_names.add(name)
            
        layer_stats.append({
            'name': name,
            'shape': list(param.shape),
            'params': param_count,
            'trainable': is_trainable,
            'dtype': str(param.dtype).split('.')[-1]
        })
    
    for name in sorted(trainable_param_names):
        is_low_level = True
        for other_name in trainable_param_names:
            if other_name != name and other_name.startswith(name + '.'):
                is_low_level = False
                break
        if is_low_level:
            trainable_layers.append(name)
    
    trainable_percent = (trainable_param_count / total_param_count * 100) if total_param_count > 0 else 0
    param_counts_df = pd.DataFrame(layer_stats)
    logger.info(f"trainable layers: {trainable_layers}")
    
    return ParameterInfo(
        total_params=total_param_count,
        trainable_params=trainable_param_count,
        trainable_percent=trainable_percent,
        trainable_layers=trainable_layers,
        param_counts_by_layer=param_counts_df
    )