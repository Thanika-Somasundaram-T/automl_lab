import json
from pathlib import Path
import random
from typing import Any
import timm


from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torch.nn as nn
from torchvision.models import (
    resnet18, ResNet18_Weights,
    resnet34, ResNet34_Weights,
    mobilenet_v2, MobileNet_V2_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
    vit_b_16, ViT_B_16_Weights
)


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    # For deterministic behavior on CuDNN backend
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def calculate_mean_std(dataset_class: Any):
    """Calculate the mean and standard deviation of the entire image dataset."""
    mean = 0.
    std = 0.
    total_images_count = 0

    dataset = dataset_class(
        root="./data",
        split='train',
        download=True,
        transform=transforms.ToTensor()
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    for images, _ in loader:
        batch_samples = images.size(0)  # batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += batch_samples

    mean /= total_images_count
    std /= total_images_count

    return mean, std


def get_data_loader(dataset_class, transform, batch_size, split="train", is_val=False, seed=42):
    dataset = dataset_class(
        root="./data",
        split=split,
        download=True,
        transform=transform
    )

    if split == "train":
        if is_val:
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = random_split(
                dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(seed)
            )
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            return train_loader, val_loader
        else:
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            return train_loader, None
    else:
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return data_loader, None


def build_model(model_name: str, num_classes: int):
    if model_name == "resnet18":
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "resnet34":
        weights = ResNet34_Weights.DEFAULT
        model = resnet34(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "mobilenet_v2":
        weights = MobileNet_V2_Weights.DEFAULT
        model = mobilenet_v2(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif model_name == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT
        model = efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif model_name == "vit_b_16":
        weights = ViT_B_16_Weights.DEFAULT
        model = vit_b_16(weights=weights)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)

    elif model_name == "deit_tiny":
        model = timm.create_model("deit_tiny_patch16_224", pretrained=True)
        model.head = nn.Linear(model.head.in_features, num_classes)

    elif model_name == "swin_t":
        model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True)
        model.head = nn.Linear(model.head.in_features, num_classes)


    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return model


def plot_neps(losses_path="neps_results/losses_log.json"):
    if not Path(losses_path).exists():
        print("No loss log found to plot.")
        return

    with open(losses_path) as f:
        all_losses = json.load(f)

    plt.figure(figsize=(8,5))
    for i, losses in enumerate(all_losses["train"]):
        plt.plot(losses, label=f"Trial {i+1}")
    plt.title("Train Loss per Epoch (All NEPS Trials)")
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("neps_results/train_loss_all_trials.png")
    plt.close()

    plt.figure(figsize=(8,5))
    for i, losses in enumerate(all_losses["val"]):
        plt.plot(losses, label=f"Trial {i+1}")
    plt.title("Val Loss per Epoch (All NEPS Trials)")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("neps_results/val_loss_all_trials.png")
    plt.close()

    print("Plots saved in neps_results/")


