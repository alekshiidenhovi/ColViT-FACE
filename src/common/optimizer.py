import bitsandbytes as bnb
import torch
from transformers.trainer_pt_utils import get_parameter_names
from common.config import OptimizerConfig


def get_adam8bit_optimizer(model: torch.nn.Module, optimizer_config: OptimizerConfig):
    decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": optimizer_config.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if n not in decay_parameters
            ],
            "weight_decay": 0.0,
        },
    ]

    adam_bnb_optim = bnb.optim.Adam8bit(
        optimizer_grouped_parameters,
        betas=(optimizer_config.adam_beta1, optimizer_config.adam_beta2),
        lr=optimizer_config.learning_rate,
    )
    return adam_bnb_optim
