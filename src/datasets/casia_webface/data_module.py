import lightning as L
import os
import random
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets.casia_webface.dataset import CASIAFaceDataset
from common.config import DatasetConfig


class CASIAFaceDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_config: DatasetConfig,
    ):
        """Initialize the FaceDataModule for handling CASIA-WebFace dataset.

        Parameters
        ----------
        config : TrainingConfig model training
        """
        super().__init__()
        self.dataset_config = dataset_config

    def setup(self, stage=None):
        identity_folders = [
            d
            for d in os.listdir(self.dataset_config.dataset_dir)
            if os.path.isdir(os.path.join(self.dataset_config.dataset_dir, d))
        ]
        random.shuffle(identity_folders)

        train_split, val_split, test_split = self.dataset_config.train_val_test_split
        n_identities = len(identity_folders)
        train_size = int(n_identities * train_split)
        val_size = int(n_identities * val_split)

        train_identities = identity_folders[:train_size]
        val_identities = identity_folders[train_size : train_size + val_size]
        test_identities = identity_folders[train_size + val_size :]

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.dataset_config.img_size, self.dataset_config.img_size)
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
            ]
        )
        self.train_dataset = CASIAFaceDataset(
            self.dataset_config.dataset_dir,
            identities=train_identities,
            transform=self.transform,
            num_negative_samples=self.dataset_config.train_num_negative_samples,
        )
        self.val_dataset = CASIAFaceDataset(
            self.dataset_config.dataset_dir,
            identities=val_identities,
            transform=self.transform,
            num_negative_samples=self.dataset_config.val_num_negative_samples,
        )
        self.test_dataset = CASIAFaceDataset(
            self.dataset_config.dataset_dir,
            identities=test_identities,
            transform=self.transform,
            num_negative_samples=self.dataset_config.test_num_negative_samples,
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.dataset_config.train_batch_size,
            shuffle=True,
            num_workers=self.dataset_config.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.dataset_config.val_batch_size,
            shuffle=False,
            num_workers=self.dataset_config.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.dataset_config.test_batch_size,
            shuffle=False,
            num_workers=self.dataset_config.num_workers,
            pin_memory=True,
        )
