import typing as T
import torch

TRAINING_SAMPLE = T.Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, str, str, T.List[str]
]
