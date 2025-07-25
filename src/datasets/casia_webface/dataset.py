import os
import typing as T
import random
from collections import defaultdict
from PIL import Image
from torch.utils.data import Dataset
from common.logger import logger
from transformers import ViTImageProcessorFast


class CASIAFaceDataset(Dataset):
    def __init__(
        self,
        dir_path: str,
        identities: T.List[str],
        processor: ViTImageProcessorFast,
        num_negative_samples: int,
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
        self.processor = processor
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

    def __getitem__(self, idx: int):
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

        nof_other_identity_images = 0
        for other_identity in other_identities:
            nof_other_identity_images += len(
                self.identity_to_image_paths[other_identity]
            )

        same_identity_paths = [
            img_path
            for img_path in self.identity_to_image_paths[identity]
            if img_path != query_image_path
        ]
        if len(same_identity_paths) == 0:
            raise ValueError(
                f"No positive images found for identity {identity}. "
                f"Please check the dataset."
            )
        positive_image_path = random.choice(same_identity_paths)

        negative_image_paths = []
        used_paths = {query_image_path, positive_image_path}
        while (
            len(negative_image_paths) < self.num_negative_samples
            or len(negative_image_paths) >= nof_other_identity_images
        ):
            neg_identity = random.choice(other_identities)
            neg_path = random.choice(self.identity_to_image_paths[neg_identity])
            if neg_path not in used_paths:
                negative_image_paths.append(neg_path)
                used_paths.add(neg_path)

        image_paths = [query_image_path, positive_image_path, *negative_image_paths]

        images = [Image.open(p).convert("RGB") for p in image_paths]
        processed_images = self.processor(images=images, return_tensors="pt")
        return (processed_images.pixel_values, image_paths)
