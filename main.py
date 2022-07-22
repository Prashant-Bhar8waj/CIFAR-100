import argparse
import json
import os
import time
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from config import TrainingConfigurations
from dataset.cifar100 import get_those_loaders
from engine import evaluate_one_epoch, train_one_epoch
from models import model_dict
from tools.utils import set_seed


def get_args_parser():
    parser = argparse.ArgumentParser("Set transformer detector", add_help=False)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument(
        "--output_dir", default="", help="path where to save, empty for no saving"
    )
    parser.add_argument(
        "--start_epoch", default=1, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--eval", action="store_true")
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser("CIFAR100 training", parents=[get_args_parser()])
    args = parser.parse_args()

    print(args)
    cfg = TrainingConfigurations()
    set_seed(cfg.seed)
    cfg.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_augs = None
    test_augs = None

    train_loader, test_loader = get_those_loaders(train_augs, test_augs, cfg.batch_size)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    criterion = nn.CrossEntropyLoss()
    for model_name, Model in model_dict.items():
        print(f"\nInitiating training for {model_name}")

        model = Model(cfg.number_classes)
        model = model.to(cfg.device)

        if cfg.optimizer == "sgd":
            optimizer = optim.SGD(
                model.parameters(), lr=cfg.learning_rate, momentum=cfg.momentum
            )
        elif cfg.optimizer == "adamw":
            optimizer = optim.AdamW(
                model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
            )
        else:
            assert False, "Unknown optimizer, change training configuration"

        if cfg.lr_scheduler:
            if cfg.lr_scheduler == "step":
                lr_scheduler = optim.lr_scheduler.StepLR(optimizer, cfg.lr_drop)
            if cfg.lr_scheduler == "cosine_annealing":
                lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=cfg.T_max
                )
            else:
                assert False, "Unknown sechduler, change training configuration"
        else:
            lr_scheduler = None

        if args.resume:
            checkpoint = torch.load(
                os.path.join(args.output_dir, model_name, f"{model_name}_last.pt"),
                map_location="cpu",
            )

            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"] + 1

        print("################ STARTING TRAINING ################")
        st = time.time()
        best_loss = np.inf
        for epoch in range(args.start_epoch, cfg.epoch + 1):
            print("Epoch #{}".format(epoch))
            train_stats = train_one_epoch(
                model,
                optimizer,
                None if cfg.lr_scheduler == "step" else lr_scheduler,
                criterion,
                train_loader,
                cfg.device,
            )

            if cfg.lr_scheduler == "step":
                lr_scheduler.step()

            test_stats = evaluate_one_epoch(model, criterion, test_loader, cfg.device)

            # storing last epoch model
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                },
                os.path.join(args.output_dir, model_name, f"{model_name}_last.pt"),
            )

            # saving best model
            if test_stats["loss"] < best_loss:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                    },
                    os.path.join(args.output_dir, model_name, f"{model_name}_best.pt"),
                )

        total_time = time.time() - st
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))



""" TODO
dataset stats mean
augumentaion (albumemtaion/ trochvision)

logging 
add models 
grad cam
"""
