from urllib.request import urlopen
from PIL import Image, ImageFile
from models.colvit import ColViT
from common.config import TrainingConfig

if __name__ == "__main__":
    img: ImageFile.ImageFile = Image.open(
        fp=urlopen(
            url="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"
        )
    )
    model_name = "vit_small_patch16_384.augreg_in21k_ft_in1k"
    new_token_dim = 256
    config = TrainingConfig(
        dataset_dir="/teamspace/studios/this_studio/ColViT-FACE/data/casia-webface/"
    )
    colvit = ColViT(config)

    trainable_params = []
    total_params = 0
    trainable_param_count = 0

    for name, param in colvit.encoder.named_parameters():
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
