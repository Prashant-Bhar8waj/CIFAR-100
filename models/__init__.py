from models.efficientnet import EfficientNetV2S
from models.mobilenet import MobileNet
from models.regnet import RegNet
from models.resnet import ResNet50, ResNet101, ResNet34
from models.squeezenet import SqueezeNet
from models.swintransformer import SwinT

model_dict = {
    "efficientnet": EfficientNetV2S,
    "mobilenet": MobileNet,
    "regnet": RegNet,
    "resnet50": ResNet50,
    "resnet34": ResNet34,
    "resnet101": ResNet101,
    "squeezenet": SqueezeNet,
    "swintransformer": SwinT,
}
