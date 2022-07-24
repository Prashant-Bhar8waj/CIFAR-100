from easydict import EasyDict
from torch import _pin_memory


class TrainingConfigurations(EasyDict):
    seed = 69
    number_classes = 100
    batch_size = 128
    num_workers = 2
    pin_memory = True
    epochs = 10
    optimizer = "sgd"
    learning_rate = 1e-4
    momentum = 0.92
    weight_decay = None
    lr_scheduler = "cosine_annealing"
    min_lr = 1e-6
    lr_drop = None
    T_max = 500
