from models import ResNet50
from tools.grad_cam import GradCAM
from dataset.cifar100 import get_those_loaders
from tools.guided_backprop import GuidedBackpropReLUModel

from tools.cam_utils.image import show_cam_on_image, deprocess_image, preprocess_image


model = ResNet50(3)
_, data = get_those_loaders(None, None, 2)

print(data.dataset)
