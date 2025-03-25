from urllib.request import urlopen
from PIL import Image, ImageFile
import torch
from src.models.colvit import VitEncoder

if __name__ == "__main__":
    img: ImageFile.ImageFile = Image.open(fp=urlopen(
        url='https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
    ))
    model_name = 'vit_small_patch16_384.augreg_in21k_ft_in1k'
    new_token_dim = 256 
    model: VitEncoder = VitEncoder(reduced_dim=new_token_dim, model_name=model_name)
    
    dummy_input = torch.randn(2, 3, 384, 384)
    output = model(dummy_input)
    print("Output shape:", output.shape)  
