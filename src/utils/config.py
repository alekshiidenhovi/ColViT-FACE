from pydantic import BaseModel, Field, field_validator
import typing as T

BASE_MODEL = T.Literal["vit_small_patch16_384.augreg_in21k_ft_in1k"]


class TrainingConfig(BaseModel):
    dataset_dir: str = Field(
        description="Path to the dataset directory",
    )
    batch_size: int = Field(
        default=32,
        ge=4,
        description="Batch size for training",
    )
    num_workers: int = Field(
        default=4, description="Number of workers for data loading", ge=1
    )
    img_size: int = Field(
        default=256,
        description="Size of the input images",
    )
    train_val_test_split: T.Tuple[float, float, float] = Field(
        default=(0.95, 0.01, 0.04),
        description="Train, validation, and test split proportions",
    )
    num_negative_samples: int = Field(
        default=7,
        description="Number of negative samples to return per anchor image",
    )
    pretrained_vit_name: BASE_MODEL = Field(
        default="vit_small_patch16_384.augreg_in21k_ft_in1k",
        description="Name of the pretrained vision transformer to use",
    )
    token_embedding_dim: int = Field(
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
    val_check_interval: int = Field(
        default=100,
        ge=10,
        description="The interval of training batched to run validation",
    )
    seed: int = Field(
        default=42,
        description="Seed for training reproducibility",
    )
    learning_rate: float = Field(
        default=1e-5, ge=0, description="Learning rate of the model"
    )
    lora_rank: int = Field(
        default=8, description="Rank of the LoRA decomposition matrix", ge=2
    )
    lora_alpha: int = Field(
        default=8, description="Scaling factor for the LoRA decomposition matrix", ge=1
    )

    @field_validator("train_val_test_split")
    def validate_split_sum(cls, v):
        if sum(v) != 1:
            raise ValueError(
                "Train, validation and test split proportions must sum to 1"
            )
        return v
