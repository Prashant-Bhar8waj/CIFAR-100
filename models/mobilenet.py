import torch
import torch.nn as nn

from torchvision import models


class mobilenet(nn.Module):
    def __init__(self, nc):
        super(mobilenet, self).__init__()

        self.model = models.mobilenet_v3(pretrained=True)
        num_ftrs = self.model.fc.in_features

        self.model.fc = nn.Linear(num_ftrs, nc)
        
    def forward(self, x):        
        out = self.model(x)

        return 