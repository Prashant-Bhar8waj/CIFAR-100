import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import TrainingConfigurations
from dataset.cifar100 import download_data
from models import model_dict
from tools.pytorch_grad_cam import GradCAM
from tools.pytorch_grad_cam.cam_utils.image import (
    deprocess_image,
    preprocess_image,
    show_cam_on_image,
)


def get_args_parser():
    parser = argparse.ArgumentParser("args for grad cam", add_help=False)

    parser.add_argument("--output_dir", default="", help="path where to save")
    parser.add_argument(
        "--model",
        choices=[
            "efficient",
            "regnet",
            "resnet50",
            "resnet34",
            "resnet101",
            "mobilenet",
            "swintransformer",
        ],
        help="model name to implement grad cam",
    )

    parser.add_argument(
        "--aug_smooth",
        action="store_true",
        help="Apply test time augmentation to smooth the CAM",
    )
    parser.add_argument(
        "--eigen_smooth",
        action="store_true",
        help="Reduce noise by taking the first principle componenet"
        "of cam_weights*activations",
    )

    return parser


def reshape_transform(tensor, height=24, width=16):
    # print(tensor.shape)
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def cam_results(model, cam_algorithm, target_layer, input_tensor, reshape_transform):
    with cam_algorithm(
        model=model,
        target_layers=target_layer,
        reshape_transform=reshape_transform,
        use_cuda=args.use_cuda,
    ) as cam:
        cam.batch_size = 32
        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=None,
            aug_smooth=args.aug_smooth,
            eigen_smooth=args.eigen_smooth,
        )

        grayscale_cam = grayscale_cam[0, :]

        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
        return cam_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser("GRAD CAM", parents=[get_args_parser()])
    args = parser.parse_args()
    args.use_cuda = torch.cuda.is_available()

    print(args)
    nc = TrainingConfigurations.number_classes

    _, dataset = download_data("dataset")
    images = dataset.data

    for k in model_dict.keys():
        model_dict[k] = model_dict[k](nc)

    target_layers = {
        "efficientnet": (
            model_dict["efficientnet"].model.features[5],
            model_dict["efficientnet"].model.features[-1],
        ),
        "mobilenet": (
            model_dict["mobilenet"].model.features[11],
            model_dict["mobilenet"].model.features[-1],
        ),
        "regnet": (
            model_dict["regnet"].model.trunk_output[2],
            model_dict["regnet"].model.trunk_output[-1],
        ),
        "resnet50": (
            model_dict["resnet50"].model.layer3,
            model_dict["resnet50"].model.layer4,
        ),
        "resnet34": (
            model_dict["resnet34"].model.layer3,
            model_dict["resnet34"].model.layer4,
        ),
        "resnet101": (
            model_dict["resnet101"].model.layer3,
            model_dict["resnet101"].model.layer4,
        ),
        "swintransformer": (
            model_dict["swintransformer"].model.features[5],
            model_dict["swintransformer"].model.features[-1],
        ),
    }

    model_infer = {args.model: model_dict[args.model]} if args.model else model_dict

    for model_name, model in model_infer.items():
        rt = reshape_transform if model_name == "swintransformer" else None
        cam_algorithm = GradCAM
        fig, axs = plt.subplots(5, 3, sharex=True, sharey=True, figsize=(40, 25))
        for i in range(5):
            im = images[i]
            axs[i, 0].imshow(im)
            axs[i, 0].axis("off")
            axs[i, 0].set_title("Image")

            rgb_img = np.float32(im) / 255
            input_tensor = preprocess_image(
                rgb_img, mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]
            )

            cam_image = cam_results(
                model, cam_algorithm, [target_layers[model_name][0]], input_tensor, rt
            )
            axs[i, 1].imshow(cam_image)
            axs[i, 1].axis("off")
            axs[i, 1].set_title("Middel layer")

            cam_image = cam_results(
                model, cam_algorithm, [target_layers[model_name][1]], input_tensor, rt
            )
            axs[i, 2].imshow(cam_image)
            axs[i, 2].axis("off")
            axs[i, 2].set_title("Last layer")

        plt.show()
