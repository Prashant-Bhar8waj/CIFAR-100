import torch
import torch.nn as nn
from torchvision import models


class SqueezeNet(nn.Module):
    def __init__(self, nc):
        super(SqueezeNet, self).__init__()

        self.model = models.squeezenet1_1(pretrained=True)
        num_ftrs = self.model.fc.in_features

        self.model.fc = nn.Linear(num_ftrs, nc)

    def forward(self, x):
        out = self.model(x)

        return out
