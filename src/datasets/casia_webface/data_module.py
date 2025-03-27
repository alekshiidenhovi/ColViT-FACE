import lightning as pl
import os
import random
from torchvision import transforms
from torch.utils.data import DataLoader
from src.datasets.casia_webface.dataset import CASIAFaceDataset


class CASIAFaceDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        img_size: int = 256,
        train_val_test_split: tuple = (0.8, 0.1, 0.1),
        num_negative_samples: int = 7,
    ):
        """Initialize the FaceDataModule for handling CASIA-WebFace dataset.

        Parameters
        ----------
        dataset_dir : str
            Root directory containing identity folders with face images
        batch_size : int, optional
            Number of samples per batch, by default 32
        num_workers : int, optional
            Number of subprocesses for data loading, by default 4
        img_size : int, optional
            Size to resize images to (both height and width), by default 256
        train_val_test_split : tuple, optional
            Ratios for train/val/test split of identities, by default (0.8, 0.1, 0.1)
        num_negative_samples : int, optional
            Number of negative samples to use per anchor-positive pair, by default 5
        """
        super().__init__()

        if sum(train_val_test_split) != 1:
            raise ValueError("Sum of train_val_test_split must be 1")

        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.num_negative_samples = num_negative_samples
        self.train_val_test_split = train_val_test_split

    def setup(self, stage=None):
        identity_folders = [
            d
            for d in os.listdir(self.dataset_dir)
            if os.path.isdir(os.path.join(self.dataset_dir, d))
        ]
        random.shuffle(identity_folders)

        train_split, val_split, test_split = self.train_val_test_split
        n_identities = len(identity_folders)
        train_size = int(n_identities * train_split)
        val_size = int(n_identities * val_split)

        train_identities = identity_folders[:train_size]
        val_identities = identity_folders[train_size : train_size + val_size]
        test_identities = identity_folders[train_size + val_size :]

        self.transform = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
            ]
        )
        self.train_dataset = CASIAFaceDataset(
            self.dataset_dir, identities=train_identities, transform=self.transform
        )
        self.val_dataset = CASIAFaceDataset(
            self.dataset_dir, identities=val_identities, transform=self.transform
        )
        self.test_dataset = CASIAFaceDataset(
            self.dataset_dir, identities=test_identities, transform=self.transform
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
