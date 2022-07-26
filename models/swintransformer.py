import torch
import torch.nn as nn
from torchvision import models


class SwinT(nn.Module):
    def __init__(self, nc):
        super(SwinT, self).__init__()

        self.model = models.swin_t(num_classes=nc)

    def forward(self, x):
        out = self.model(x)

        return out
