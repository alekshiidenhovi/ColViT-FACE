import os
import random
import typing as T

from torch.utils.data import DataLoader
from datasets.casia_webface.dataset import CASIAFaceDataset
from common.config import DatasetConfig
from transformers import ViTImageProcessorFast


def retrieve_dataloaders(
    processor: ViTImageProcessorFast,
    dataset_config: DatasetConfig,
) -> T.Tuple[DataLoader, DataLoader, DataLoader]:
    identity_folders = [
        d
        for d in os.listdir(dataset_config.dataset_dir)
        if os.path.isdir(os.path.join(dataset_config.dataset_dir, d))
    ]
    random.shuffle(identity_folders)

    train_split, val_split, _ = dataset_config.train_val_test_split
    n_identities = len(identity_folders)
    train_size = int(n_identities * train_split)
    val_size = int(n_identities * val_split)

    train_identities = identity_folders[:train_size]
    val_identities = identity_folders[train_size : train_size + val_size]
    test_identities = identity_folders[train_size + val_size :]

    train_dataset = CASIAFaceDataset(
        dataset_config.dataset_dir,
        identities=train_identities,
        processor=processor,
        num_negative_samples=dataset_config.train_num_negative_samples,
    )

    val_dataset = CASIAFaceDataset(
        dataset_config.dataset_dir,
        identities=val_identities,
        processor=processor,
        num_negative_samples=dataset_config.val_num_negative_samples,
    )

    test_dataset = CASIAFaceDataset(
        dataset_config.dataset_dir,
        identities=test_identities,
        processor=processor,
        num_negative_samples=dataset_config.test_num_negative_samples,
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=dataset_config.train_batch_size,
        shuffle=True,
        num_workers=dataset_config.num_workers,
        pin_memory=True,
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=dataset_config.val_batch_size,
        shuffle=True,
        num_workers=dataset_config.num_workers,
        pin_memory=True,
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=dataset_config.test_batch_size,
        shuffle=True,
        num_workers=dataset_config.num_workers,
        pin_memory=True,
    )

    return train_dataloader, val_dataloader, test_dataloader
