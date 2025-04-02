import torch
from urllib.request import urlopen
from PIL import Image, ImageFile
from models.vit_encoder_with_lora import VitEncoderWithLoRA
from transformers import ViTModel, AutoConfig, BitsAndBytesConfig
from common.config import ModelConfig

if __name__ == "__main__":
    img: ImageFile.ImageFile = Image.open(
        fp=urlopen(
            url="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"
        )
    )
    model_name = "google/vit-base-patch16-224"
    new_token_dim = 256
    model_config = ModelConfig(
        pretrained_vit_name=model_name, token_embedding_dim=new_token_dim
    )
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    vit_config = AutoConfig.from_pretrained(model_config.pretrained_vit_name)
    model = VitEncoderWithLoRA(
        vit_config=vit_config,
        model_config=model_config,
        quantization_config=quantization_config,
    )

    trainable_params = []
    total_params = 0
    trainable_param_count = 0

    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params.append(name)
            trainable_param_count += param.numel()

    print("Trainable parameters:")
    for name in trainable_params:
        print(f"  - {name}")

    print(f"\nTotal parameters: {total_params:,}")
    print(
        f"Trainable parameters: {trainable_param_count:,} ({trainable_param_count / total_params:.2%})"
    )
