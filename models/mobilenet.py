import torch
import torch.nn as nn
from torchvision import models


class MobileNet(nn.Module):
    def __init__(self, nc):
        super(MobileNet, self).__init__()

        self.model = models.mobilenet_v3_large(pretrained=True)

        num_ftrs = self.model.classifier[3].in_features
        self.model.classifier[3] = nn.Linear(num_ftrs, nc)

    def forward(self, x):
        out = self.model(x)

        return out
