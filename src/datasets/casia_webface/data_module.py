import lightning as pl
import os
import random
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets.casia_webface.dataset import CASIAFaceDataset
from utils.config import TrainingConfig


class CASIAFaceDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config: TrainingConfig,
    ):
        """Initialize the FaceDataModule for handling CASIA-WebFace dataset.

        Parameters
        ----------
        config : TrainingConfig model training
        """
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        identity_folders = [
            d
            for d in os.listdir(self.config.dataset_dir)
            if os.path.isdir(os.path.join(self.config.dataset_dir, d))
        ]
        random.shuffle(identity_folders)

        train_split, val_split, test_split = self.config.train_val_test_split
        n_identities = len(identity_folders)
        train_size = int(n_identities * train_split)
        val_size = int(n_identities * val_split)

        train_identities = identity_folders[:train_size]
        val_identities = identity_folders[train_size : train_size + val_size]
        test_identities = identity_folders[train_size + val_size :]

        self.transform = transforms.Compose(
            [
                transforms.Resize((self.config.img_size, self.config.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
            ]
        )
        self.train_dataset = CASIAFaceDataset(
            self.config.dataset_dir,
            identities=train_identities,
            transform=self.transform,
            num_negative_samples=self.config.num_negative_samples,
        )
        self.val_dataset = CASIAFaceDataset(
            self.config.dataset_dir,
            identities=val_identities,
            transform=self.transform,
            num_negative_samples=self.config.num_negative_samples,
        )
        self.test_dataset = CASIAFaceDataset(
            self.config.dataset_dir,
            identities=test_identities,
            transform=self.transform,
            num_negative_samples=self.config.num_negative_samples,
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )
