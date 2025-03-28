import os
import typing as T
import random
import torch
from collections import defaultdict
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from src.logger import logger
from src.types import TRAINING_SAMPLE


class CASIAFaceDataset(Dataset):
    def __init__(
        self,
        dir_path: str,
        identities: T.List[str],
        transform: transforms.Compose,
        num_negative_samples: int = 5,
    ):
        """Initialize the FaceTripletDataset.

        Parameters
        ----------
        dir_path : str
            Root directory containing identity folders with face images
        identities : List[str]
            List of identity folder names to include in this dataset
        transform : callable
            Transforms to apply to the images
        num_negative_samples : int, optional
            Number of negative samples to return per anchor image, by default 5

        """
        self.dir_path = dir_path
        self.transform = transform
        self.identity_to_image_paths: T.Dict[str, T.List[str]] = defaultdict(list)
        self.image_paths: T.List[str] = []
        self.image_identities: T.List[str] = []
        self.num_negative_samples = num_negative_samples

        for identity in identities:
            identity_path = os.path.join(dir_path, identity)
            if os.path.isdir(identity_path):
                for img_name in os.listdir(identity_path):
                    if img_name.lower().endswith(".jpg"):
                        img_path = os.path.join(identity_path, img_name)
                        self.image_paths.append(img_path)
                        self.image_identities.append(identity)
                        self.identity_to_image_paths[identity].append(img_path)
                    else:
                        logger.warning(f"Unsupported image data type: {img_name}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> TRAINING_SAMPLE:
        """Get an image with its negative samples.

        Parameters
        ----------
        idx : int
            Index of the image in the dataset

        Returns
        -------
        tuple
            Contains (image, list of negative_images, identity)
            where image is the anchor image,
            negative_images is a list of randomly sampled images from other identities,
            and identity is the identity label of the anchor image
        """
        query_image_path = self.image_paths[idx]
        identity = self.image_identities[idx]
        other_identities = [
            id for id in self.identity_to_image_paths.keys() if id != identity
        ]

        positive_image_path = random.choice(
            [
                img_path
                for img_path in self.identity_to_image_paths[identity]
                if img_path != query_image_path
            ]
        )

        negative_image_paths = []
        for _ in range(self.num_negative_samples):
            neg_identity = random.choice(other_identities)
            neg_path = random.choice(self.identity_to_image_paths[neg_identity])
            negative_image_paths.append(neg_path)

        query_image: torch.Tensor = self.transform(
            Image.open(query_image_path).convert("RGB")
        )
        positive_image: torch.Tensor = self.transform(
            Image.open(positive_image_path).convert("RGB")
        )
        negative_imgs = [
            self.transform(Image.open(p).convert("RGB")) for p in negative_image_paths
        ]
        negative_imgs = torch.stack(negative_imgs)

        return (
            query_image.unsqueeze(0),
            positive_image.unsqueeze(0),
            negative_imgs,
            query_image_path,
            positive_image_path,
            negative_image_paths,
        )
