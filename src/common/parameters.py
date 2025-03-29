import torch
import typing as T


def count_parameters(model: torch.nn.Module) -> T.Tuple[int, int]:
    """Count total and trainable parameters in a PyTorch model.

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model to analyze

    Returns
    -------
    tuple
        A tuple containing:
        - total_param_count (int): Total number of parameters in the model
        - trainable_param_count (int): Number of trainable parameters in the model
    """
    total_param_count = 0
    trainable_param_count = 0

    for name, param in model.named_parameters():
        total_param_count += param.numel()
        if param.requires_grad:
            trainable_param_count += param.numel()

    return total_param_count, trainable_param_count
