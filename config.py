from easydict import EasyDict
from torch import _pin_memory


class TrainingConfigurations(EasyDict):
    seed = 69
    number_classes = 100
    batch_size = 128
    num_workers = 1
    pin_memory = False
    epochs = 10
    optimizer = "sgd"
    learning_rate = 1e-5
    momentum = 0.9
    weight_decay = None
    lr_scheduler = "steplr"
    lr_drop = None
    T_max = None
