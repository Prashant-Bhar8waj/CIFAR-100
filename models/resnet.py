import torch
import torch.nn as nn

from torchvision import models


class ResNet50(nn.Module):
    def __init__(self, nc):
        super(ResNet50, self).__init__()

        self.model = models.resnet50(pretrained=True)
        num_ftrs = self.model.fc.in_features

        self.model.fc = nn.Linear(num_ftrs, nc)
        
    def forward(self, x):        
        out = self.model(x)

        return 
        

class ResNet30(nn.Module):
    def __init__(self, nc):
        super(ResNet30, self).__init__()

        self.model = models.resnet30(pretrained=True)
        num_ftrs = self.model.fc.in_features

        self.model.fc = nn.Linear(num_ftrs, nc)
        
    def forward(self, x):        
        out = self.model(x)

        return out


class ResNet18(nn.Module):
    def __init__(self, nc):
        super(ResNet18, self).__init__()

        self.model = models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features

        self.model.fc = nn.Linear(num_ftrs, nc)
        
    def forward(self, x):        
        out = self.model(x)

        return out

