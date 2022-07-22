from easydict import EasyDict


class TrainingConfigurations(EasyDict):
    seed = 69
    number_classes = 100
    batch_size = 128
    epochs = 10
    optimizer = "sgd"
    learning_rate = 1e-5
    momentum = 0.9
    weight_decay = None
    lr_scheduler = "steplr"
    lr_drop = None
    T_max = None
