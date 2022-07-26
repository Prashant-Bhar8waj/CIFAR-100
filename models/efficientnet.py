import torch
import torch.nn as nn
from torchvision import models


class EfficientNetV2S(nn.Module):
    def __init__(self, nc):
        super(EfficientNetV2S, self).__init__()

        self.model = models.efficientnet_v2_s(pretrained=True)

        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_ftrs, nc)

    def forward(self, x):
        out = self.model(x)

        return out
