from easydict import EasyDict


class TrainingConfigurations(EasyDict):
    seed = 69
    number_classes = 100
    bs = 64
    num_workers = 2
    pin_memory = True
    epochs = 40
    optimizer = "sgd"
    learning_rate = 1e-3
    momentum = 0.92
    weight_decay = 0
    lr_scheduler = "cosine_annealing"
    min_lr = 1e-2
    lr_drop = None
    T_max = 500
    patience = 6
