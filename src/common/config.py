from pydantic import BaseModel, Field, field_validator
import typing as T
import multiprocessing
from accelerate.utils.dataclasses import PrecisionType
from wandb_logger import init_wandb_api_client

BASE_MODEL = T.Literal["google/vit-base-patch16-224"]
ACCELERATOR = T.Literal["gpu", "cpu", "tpu"]
FINETUNING_MODE = T.Literal["full", "final", "final+lora"]
BENCHMARK_TYPE = T.Literal["vector_index", "full_rerank"]

class BenchmarkConfig(BaseModel):
    model_dir: str = Field(description="Path to model checkpoint directory")
    lfw_dataset_dir: str = Field(description="Path to LFW dataset directory")
    max_images_per_identity: int = Field(description="Maximum number of images per identity")
    benchmark_type: BENCHMARK_TYPE = Field(description="Type of benchmark to run")


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
        ge=1,
        description="Batch size for training",
    )
    val_batch_size: int = Field(
        default=8,
        ge=1,
        description="Batch size for validation",
    )
    test_batch_size: int = Field(
        default=1,
        ge=1,
        description="Batch size for testing",
    )
    num_workers: int = Field(
        default=multiprocessing.cpu_count() - 2,
        description="Number of workers for data loading",
        ge=1,
    )
    img_size: int = Field(
        default=224,
        ge=32,
        description="Size of the input images",
    )
    train_val_test_split: T.Tuple[float, float, float] = Field(
        default=(0.95, 0.005, 0.045),
        description="Train, validation, and test split proportions",
    )
    train_num_negative_samples: int = Field(
        default=63,
        ge=1,
        description="Number of negative samples to return per anchor image during training",
    )
    val_num_negative_samples: int = Field(
        default=63,
        ge=1,
        description="Number of negative samples to return per anchor image during validation",
    )
    test_num_negative_samples: int = Field(
        default=1000,
        ge=1,
        description="Number of negative samples to return per anchor image during testing",
    )

    @field_validator("train_val_test_split")
    def validate_split_sum(cls, v):
        ratio = sum(v)
        if not abs(ratio - 1.0) < 1e-6:
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
        default="google/vit-base-patch16-224",
        description="Name of the pretrained vision transformer to use",
    )
    token_embedding_dim: int = Field(
        default=128, ge=1, description="Final dimension of the token embeddings"
    )


class LoraParamConfig(BaseModel):
    """Configuration for LoRA parameters.

    Defines parameters for Low-Rank Adaptation (LoRA) including rank, scaling factor,
    target modules to adapt, and bias settings.
    """

    lora_rank: int = Field(
        default=16, description="Rank of the LoRA decomposition matrix", ge=2
    )
    lora_alpha: int = Field(
        default=4, description="Scaling factor for the LoRA decomposition matrix", ge=1
    )
    lora_target_modules: T.List[str] = Field(
        default=["query", "key", "value", "dense"],
        description="Modules to apply LoRA to",
    )
    lora_bias: T.Literal['none', 'all', 'lora_only'] = Field(
        default="none",
        description="Bias type for LoRA",
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
    enable_checkpointing: bool = Field(
        default=True, description="Enables checkpointing of the latest training epoch"
    )
    val_check_interval: int = Field(
        default=200,
        ge=1,
        description="The interval of training batched to run validation",
    )
    limit_val_batches: int = Field(
        default=20, description="Number of validation batches to run", ge=1
    )
    precision: PrecisionType = Field(
        default=PrecisionType.BF16,
        description="Precision of gradients and the model used during finetuning",
    )
    model_checkpoint_path: str = Field(
        default="./.checkpoints/",
        description="Folder for saving model checkpoints during training",
    )
    final_model_path: str = Field(
        default="./.checkpoints/final_model/",
    )
    max_epochs: int = Field(
        default=8,
        ge=1,
        description="Maximum number of epochs to train for",
    )
    devices: T.List[int] = Field(
        default=[0],
        description="GPU devices to use for training",
    )
    accumulate_grad_batches: int = Field(
        default=1,
        ge=1,
        description="Number of batches to accumulate gradients before updating the model",
    )
    gradient_checkpointing: bool = Field(
        default=False,
        description="Enables gradient checkpointing for memory efficient training",
    )
    finetuning_mode: FINETUNING_MODE = Field(
        default="final+lora",
        description="Mode of finetuning to use, 'full' for full finetuning, 'final' for final layer finetuning, 'final+lora' final layer regular finetuning + original layer finetuning with LoRA",
    )


class OptimizerConfig(BaseModel):
    """Configuration for the optimizer.

    Contains parameters for the optimizer including weight decay, beta1 and beta2 parameters for the Adam optimizer.
    """

    weight_decay: float = Field(
        default=0.01,
        ge=0,
        description="Weight decay for the optimizer",
    )
    adam_beta1: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Beta1 parameter for the Adam optimizer, used for the first moment estimate",
    )
    adam_beta2: float = Field(
        default=0.999,
        ge=0.0,
        le=1.0,
        description="Beta2 parameter for the Adam optimizer, used for the second moment estimate",
    )
    learning_rate: float = Field(
        default=1e-5, ge=0, description="Learning rate of the model"
    )


class TrainingConfig(
    DatasetConfig, ModelConfig, FinetuningConfig, OptimizerConfig, LoraParamConfig
):
    """Complete training configuration combining dataset, model and fine-tuning settings.

    Inherits from DatasetConfig, ModelConfig, FinetuningConfig and OptimizerConfig to provide a comprehensive configuration for the entire training pipeline.
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

    def get_optimizer_config(self) -> OptimizerConfig:
        """Get optimizer configuration."""
        return OptimizerConfig(
            **{
                k: v
                for k, v in self.model_dump().items()
                if k in OptimizerConfig.model_fields
            }
        )

    def get_lora_param_config(self) -> LoraParamConfig:
        """Get LoRA configuration."""
        return LoraParamConfig(
            **{
                k: v
                for k, v in self.model_dump().items()
                if k in LoraParamConfig.model_fields
            }
        )
        
    def load_from_wandb(self, run_id: str):
        """Load the configuration from a W&B run."""
        wandb_api = init_wandb_api_client()
        run = wandb_api.run(run_id)
        valid_config = {k: v for k, v in run.config.items() if k in self.model_fields}
        return self(**valid_config)
