import albumentations
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


def download_data(path):
    """Download CIFAR10 dataset form torchvision datasets
    Args:
        path (str, optional): download path for dataset
    Returns:
        torchvision instance: training and testing set
    """
    trainset = datasets.CIFAR100(root=path, train=True, download=True)
    testset = datasets.CIFAR100(root=path, train=False, download=True)

    print("Downloaded CIFAR100 to", path)
    return trainset, testset


class LoadDataset(Dataset):
    """Torch Dataset instance for loading dataset"""

    def __init__(self, data, transform=False):
        self.data = data
        self.aug = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        image, label = self.data[i]

        # apply augmentation only for training
        if isinstance(self.aug, albumentations.Compose):
            image = self.aug(image=np.array(image.convert("RGB")))["image"]
        elif isinstance(self.aug, transforms.Compose):
            image = self.aug(image.convert("RGB"))

        return image, label


def get_those_loaders(train_transforms, test_transforms, cfg, download_path="dataset"):
    """Generate Torch instance for Train and Test data loaders
    Args:
        train_transforms (albumentations compose class): training tansformations to be applied over images
        test_transforms (albumentations compose class): testing tansformations to be applied over images
        cfg (easydict): Batch size, pin memory, num workers to be used
        download_path (str): download path for dataset. Defaults to '/content/data'. (For Google Colab)
    Returns:
        torch instace: train and test data loaders
    """

    trainset, testset = download_data(download_path)

    train_loader = DataLoader(
        LoadDataset(trainset, train_transforms),
        batch_size=cfg.bs,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    test_loader = DataLoader(
        LoadDataset(testset, test_transforms),
        batch_size=cfg.bs,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

    print("Train & Test Loaders Created")
    return train_loader, test_loader
