import json
import os
import random

import numpy as np
import torch


def set_seed(seed):
    # for REPRODUCIBILITY.
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def datastats(loader):
    """Calculate mean and standard deviation of the dataset
    Args:
        loader (instance): torch instance for data loader
    Returns:
        tensor: mean and std of data
    """
    channel_sum, channel_squared_sum, num_batches = 0, 0, 0

    for img, _ in loader:
        channel_sum += torch.mean(img, dim=[0, 2, 3])
        channel_squared_sum += torch.mean((img) ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channel_sum / num_batches
    std = (channel_squared_sum / num_batches - mean**2) ** 0.5
    print("The mean of dataset : ", mean)
    print("The std of dataset : ", std)
    return mean, std


def log_history(history, path):
    with open(path, "a+") as f:
        f.write(json.dumps(history, indent=4))

