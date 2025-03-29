from urllib.request import urlopen
from PIL import Image, ImageFile
from models.vit_encoder import VitEncoder

if __name__ == "__main__":
    img: ImageFile.ImageFile = Image.open(
        fp=urlopen(
            url="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"
        )
    )
    model_name = "vit_small_patch16_384.augreg_in21k_ft_in1k"
    new_token_dim = 256
    model: VitEncoder = VitEncoder(reduced_dim=new_token_dim, model_name=model_name)
    model.train()
    print(model)
