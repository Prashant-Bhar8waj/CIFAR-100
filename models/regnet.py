import torch
import torch.nn as nn
from torchvision import models


class RegNet(nn.Module):
    def __init__(self, nc):
        super(RegNet, self).__init__()

        self.model = models.regnet_y_400mf(pretrained=True)
        num_ftrs = self.model.fc.in_features

        self.model.fc = nn.Linear(num_ftrs, nc)

    def forward(self, x):
        out = self.model(x)

        return out
