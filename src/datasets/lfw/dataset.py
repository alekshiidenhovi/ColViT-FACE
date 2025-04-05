import os
import random
import typing as T
from collections import defaultdict
from PIL import Image
from torch.utils.data import Dataset
from common.logger import logger
from transformers import ViTImageProcessorFast


class LFWBenchmarkDataset(Dataset):
    def __init__(
        self,
        dir_path: str,
        processor: ViTImageProcessorFast,
        image_identities: T.List[str],
        image_paths: T.List[str],
        
    ):
        """Initialize the LFW Benchmark Dataset.
        
        Parameters
        ----------
        dir_path : str
            Root directory containing identity folders with face images
        processor : ViTImageProcessorFast
            Image processor for the model
        max_images_per_identity : int
            Maximum number of images to use per identity
        """
        self.dir_path = dir_path
        self.processor = processor
        self.image_paths = image_paths
        self.image_identities = image_identities

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Get an image by index.
        
        Parameters
        ----------
        idx : int
            Index of the image
            
        Returns
        -------
        tuple
            Contains (processed_image, image_path, identity)
        """
        image_path = self.image_paths[idx]
        identity = self.image_identities[idx]
        
        image = Image.open(image_path).convert("RGB")
        processed_image = self.processor(images=image, return_tensors="pt")
        
        return (processed_image.pixel_values, image_path, identity)