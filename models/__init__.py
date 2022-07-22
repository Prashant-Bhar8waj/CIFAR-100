from models.inception import Inception
from models.mobilenet import MobileNet
from models.regnet import RegNet
from models.resnet import ResNet50, ResNet101, ResNet34
from models.squeezenet import SqueezeNet


model_dict = {
    "inception": Inception,
    "regnet": RegNet,
    "resnet50": ResNet50,
    "resnet34": ResNet34,
    "resnet101": ResNet101,
    "squeezenet": SqueezeNet,
    "mobilenet": MobileNet,
}
