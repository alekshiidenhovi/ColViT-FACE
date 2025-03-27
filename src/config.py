from pydantic import BaseModel, Field
import typing as T

BASE_MODEL = T.Literal["vit_small_patch16_384.augreg_in21k_ft_in1k"]


class TrainingConfig(BaseModel):
    pretrained_vit_name: BASE_MODEL = Field(
        default="vit_small_patch16_384.augreg_in21k_ft_in1k",
        description="Name of the pretrained vision transformer to use",
    )
    dim: int = Field(
        default=128, ge=1, description="Final dimension of the token embeddings"
    )
    learning_rate: float = Field(
        default=3e-5, ge=0, description="Learning rate of the model"
    )
    warmup_steps: int = Field(
        default=1000, ge=0, description="Number of steps to warmup the learning rate"
    )
    max_steps: int = Field(
        description="Maximum number of steps to train the model", ge=0, default=10000
    )
    enable_checkpointing: bool = Field(
        default=True, description="Enables checkpointing of the latest training epoch"
    )
