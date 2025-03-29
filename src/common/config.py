from pydantic import BaseModel, Field, field_validator
import typing as T

BASE_MODEL = T.Literal["vit_small_patch16_384.augreg_in21k_ft_in1k"]
ACCELERATOR = T.Literal["gpu", "cpu", "tpu"]
PRECISION = T.Literal[
    "bf16-mixed",
    "32",
]


class DatasetConfig(BaseModel):
    """Configuration for dataset loading and preprocessing.

    Contains parameters for data loading, batch sizes, image dimensions,
    dataset splits, and sampling configurations for training, validation and testing.
    """

    dataset_dir: str = Field(
        description="Path to the dataset directory",
    )
    train_batch_size: int = Field(
        default=8,
        ge=4,
        description="Batch size for training",
    )
    val_batch_size: int = Field(
        default=4,
        description="Batch size for validation",
    )
    test_batch_size: int = Field(
        default=1,
        description="Batch size for testing",
    )
    num_workers: int = Field(
        default=8, description="Number of workers for data loading", ge=1
    )
    img_size: int = Field(
        default=384,
        description="Size of the input images",
    )
    train_val_test_split: T.Tuple[float, float, float] = Field(
        default=(0.95, 0.01, 0.04),
        description="Train, validation, and test split proportions",
    )
    train_num_negative_samples: int = Field(
        default=7,
        description="Number of negative samples to return per anchor image during training",
    )
    val_num_negative_samples: int = Field(
        default=63,
        description="Number of negative samples to return per anchor image during validation",
    )
    test_num_negative_samples: int = Field(
        default=1000,
        description="Number of negative samples to return per anchor image during testing",
    )

    @field_validator("train_val_test_split")
    def validate_split_sum(cls, v):
        if sum(v) != 1:
            raise ValueError(
                "Train, validation and test split proportions must sum to 1"
            )
        return v


class ModelConfig(BaseModel):
    """Configuration for the model architecture.

    Defines parameters for the vision transformer model including the pretrained model name
    and embedding dimensions.
    """

    pretrained_vit_name: BASE_MODEL = Field(
        default="vit_small_patch16_384.augreg_in21k_ft_in1k",
        description="Name of the pretrained vision transformer to use",
    )
    token_embedding_dim: int = Field(
        default=128, ge=1, description="Final dimension of the token embeddings"
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


class FinetuningConfig(BaseModel):
    """Configuration for model fine-tuning.

    Contains training hyperparameters including learning rate schedules, validation intervals,
    checkpointing settings and LoRA-specific parameters.
    """

    accelerator: ACCELERATOR = Field(
        default="gpu",
        description="Compute accelerator to use for training",
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
    precision: PRECISION = Field(
        default="bf16-mixed",
        description="Precision of gradients and the model used during finetuning",
    )


class TrainingConfig(DatasetConfig, ModelConfig, FinetuningConfig):
    """Complete training configuration combining dataset, model and fine-tuning settings.

    Inherits from DatasetConfig, ModelConfig and FinetuningConfig to provide a comprehensive
    configuration for the entire training pipeline, including an additional seed parameter
    for reproducibility.
    """

    seed: int = Field(
        default=42,
        description="Seed for training reproducibility",
    )

    def get_dataset_config(self) -> DatasetConfig:
        """Get dataset-specific configuration."""
        return DatasetConfig(
            **{
                k: v
                for k, v in self.model_dump().items()
                if k in DatasetConfig.model_fields
            }
        )

    def get_model_config(self) -> ModelConfig:
        """Get model architecture configuration."""
        return ModelConfig(
            **{
                k: v
                for k, v in self.model_dump().items()
                if k in ModelConfig.model_fields
            }
        )

    def get_finetuning_config(self) -> FinetuningConfig:
        """Get fine-tuning hyperparameters configuration."""
        return FinetuningConfig(
            **{
                k: v
                for k, v in self.model_dump().items()
                if k in FinetuningConfig.model_fields
            }
        )
