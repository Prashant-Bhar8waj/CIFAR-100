import numpy as np

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets


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
    """Torch Dataset instance for loadinf dataset and transforming it"""

    def __init__(self, data, transform=False):
        self.data = data
        self.aug = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        image, label = self.data[i]

        # apply augmentation only for training
        if self.aug:
            image = self.aug(image=np.array(image))["image"]

        return image, label


def get_those_loaders(train_transforms, test_transforms, bs, download_path="dataset"):
    """Generate Torch instance for Train and Test data loaders
    Args:
        train_transforms (albumentations compose class): training tansformations to be applied over images
        test_transforms (albumentations compose class): testing tansformations to be applied over images
        bs (int): Batch size to be used
        download_path (str): download path for dataset. Defaults to '/content/data'. (For Google Colab)
    Returns:
        torch instace: train and test data loaders
    """

    trainset, testset = download_data(download_path)

    train_loader = DataLoader(
        LoadDataset(trainset, train_transforms),
        batch_size=bs,
        shuffle=True,
        num_workers=2,
    )
    test_loader = DataLoader(
        LoadDataset(testset, test_transforms),
        batch_size=bs,
        shuffle=False,
        num_workers=1,
    )

    print("Train & Test Loaders Created")
    return train_loader, test_loader
