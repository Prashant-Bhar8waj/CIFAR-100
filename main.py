import argparse
import json
import os
import time
import datetime

import albumentations as A
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from albumentations.pytorch import ToTensorV2
from torchvision import transforms

from config import TrainingConfigurations
from dataset.cifar100 import get_those_loaders
from engine import evaluate_one_epoch, train_one_epoch
from models import model_dict
from tools.utils import set_seed, datastats, store_stats


def get_args_parser():
    parser = argparse.ArgumentParser("Set transformer detector", add_help=False)
    parser.add_argument("--resume", action="store_true", help="resume from checkpoint")
    parser.add_argument("--output_dir", default="", help="path where to save")
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
    print("Using Device", cfg.device)

    aug = transforms.Compose([transforms.ToTensor()])
    exp_dataloader, _ = get_those_loaders(aug, aug, cfg)
    mean, std = datastats(exp_dataloader)

    train_augs = A.Compose(
        [
            A.Rotate(
                limit=5,
                interpolation=1,
                border_mode=4,
                value=None,
                mask_value=None,
                always_apply=False,
                p=0.5,
            ),
            A.Sequential(
                [
                    A.CropAndPad(px=4, keep_size=False),
                    A.RandomCrop(32, 32),
                ]
            ),
            A.CoarseDropout(
                1,8,8, 1, 8, 8, fill_value=mean, mask_fill_value=None
            ),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )
    test_augs = A.Compose([A.Normalize(mean=mean, std=std), ToTensorV2()])

    train_dataloader, test_dataloader = get_those_loaders(train_augs, test_augs, cfg)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    history = {}
    if args.resume:
        history = json.load(
            open(os.path.join(args.output_dir, "training_log.json"), "r")
        )

    init_st = time.time()
    criterion = nn.CrossEntropyLoss()
    # model ------------------------------------------------------------------
    for model_name, Model in model_dict.items():
        if (
            args.resume
            and model_name in history.keys()
            and history[model_name].get("completed")
        ):
            print("Already trained skipping ", model_name)
            continue
        elif not args.resume or model_name not in history.keys():
            history[model_name] = {"train": [], "test": []}

        # print(f"\nInitiating training for {model_name}")

        if not os.path.exists(os.path.join(args.output_dir, model_name)):
            os.mkdir(os.path.join(args.output_dir, model_name))

        model = Model(cfg.number_classes)
        model = model.to(cfg.device)

        if cfg.optimizer == "sgd":
            optimizer = optim.SGD(
                model.parameters(),
                lr=cfg.learning_rate,
                momentum=cfg.momentum,
                weight_decay=cfg.weight_decay,
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
            elif cfg.lr_scheduler == "cosine_annealing":
                lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=cfg.T_max, eta_min=cfg.min_lr
                )
            elif cfg.lr_scheduler == "reducelronpleatue":
                lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", patience=8, verbose=True
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
            best_checkpoint = torch.load(
                os.path.join(args.output_dir, model_name, f"{model_name}_best.pt"),
                map_location="cpu",
            )

            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"]

        print(optimizer, lr_scheduler)
        print(f"\n-------------- STARTING TRAINING {model_name} --------------")
        st = time.time()
        best_loss = best_checkpoint["best_loss"] if args.resume else np.inf
        # epoch ------------------------------------------------------------------
        for epoch in range(args.start_epoch, cfg.epochs + 1):
            print("Epoch #{}".format(epoch))

            train_stats = train_one_epoch(
                model,
                optimizer,
                None if cfg.lr_scheduler == "step" else lr_scheduler,
                criterion,
                train_dataloader,
                cfg.device,
            )

            if cfg.lr_scheduler == "step":
                lr_scheduler.step()

            test_stats = evaluate_one_epoch(
                model, criterion, test_dataloader, cfg.device
            )

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
                print(
                    "Validation Loss Improved (%g ---> %g)"
                    % (test_stats["loss"], best_loss)
                )
                best_loss = test_stats["loss"]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "best_loss": best_loss,
                        "epoch": epoch,
                    },
                    os.path.join(args.output_dir, model_name, f"{model_name}_best.pt"),
                )
            else:
                print("Validation loss did not improve from ", best_loss)

            history[model_name]["train"].append(
                {
                    "epoch": epoch,
                    "accuracy": train_stats["accuracy"],
                    "loss": train_stats["loss"],
                }
            )
            history[model_name]["test"].append(
                {
                    "epoch": epoch,
                    "accuracy": test_stats["accuracy"],
                    "loss": test_stats["loss"],
                }
            )

            store_stats(history, os.path.join(args.output_dir, "training_log.json"))
            # end epoch ----------------------------------------------------------------------------------------------------

        history[model_name]["completed"] = True
        store_stats(history, os.path.join(args.output_dir, "training_log.json"))

        print("\nDone training for ", model_name)
        total_time = time.time() - st
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))

        # end training ------------------------------------------------------------------
    print("\n\nDone training for all models")
    total_time = time.time() - init_st
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


""" TODO

add models 
grad cam
"""
