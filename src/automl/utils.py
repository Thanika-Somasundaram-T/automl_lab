import json
from pathlib import Path
import random
from typing import Any
import timm


from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms
import torch.nn as nn
from torchvision.models import (
    resnet50, ResNet50_Weights,
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


from torch.utils.data import Subset

def get_data_loader(
    dataset_class,
    batch_size,
    train_transform=None,
    split="train",
    is_val=False,
    seed=42,
    val_transform=None,
    test_transform=None
):
    if split == "train":
        # Load full dataset without transform (we'll assign transforms later)
        full_dataset = dataset_class(
            root="./data",
            split=split,
            download=True,
            transform=None
        )

        if is_val:
            # Calculate split sizes
            train_size = int(0.8 * len(full_dataset))
            val_size = len(full_dataset) - train_size

            # Generate indices for train and val splits
            generator = torch.Generator().manual_seed(seed)
            indices = torch.randperm(len(full_dataset), generator=generator)

            train_indices = indices[:train_size].tolist()
            val_indices = indices[train_size:].tolist()

            # Create datasets with their own transforms
            train_dataset = Subset(
                dataset_class(
                    root="./data",
                    split=split,
                    download=True,
                    transform=train_transform
                ),
                train_indices
            )
            val_dataset = Subset(
                dataset_class(
                    root="./data",
                    split=split,
                    download=True,
                    transform=val_transform
                ),
                val_indices
            )

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            return train_loader, val_loader
        else:
            # No val split, just use full dataset with train transform
            dataset = dataset_class(
                root="./data",
                split=split,
                download=True,
                transform=train_transform
            )
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            return train_loader, None

    else:
        # For val/test splits
        dataset = dataset_class(
            root="./data",
            split=split,
            download=True,
            transform=test_transform
        )
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

    elif model_name == "swin_tiny":
        print("in==================================================================")
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


# model_utils.py (extended)

import torch
import torch.nn as nn
from torchvision import models

def get_model(model_name: str, num_classes: int) -> nn.Module:
    if model_name == 'resnet50':
        weights = ResNet50_Weights.DEFAULT  # Using default weights for resnet50
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == 'efficientnet_b0':
        weights = EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif model_name == 'convnext_tiny':
        model = timm.create_model("convnext_tiny", pretrained=True)
        model.head.fc = nn.Linear(model.head.fc.in_features, num_classes)

    elif model_name == 'swin_tiny':
        model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True)

        # ---- 3. Correctly replace the head ----
        class SwinClassifierHead(nn.Module):
            def __init__(self, in_features, num_classes):
                super().__init__()
                self.fc = nn.Linear(in_features, num_classes)

            def forward(self, x):
                # x is [B, H, W, C] → flatten to [B, N, C]
                B, H, W, C = x.shape
                x = x.view(B, H * W, C)   # flatten spatial dimensions
                x = x.mean(dim=1)         # global average over tokens
                x = self.fc(x)            # [B, num_classes]
                return x

        num_classes = 10
        model.head = SwinClassifierHead(model.head.fc.in_features, num_classes)
    else:
        raise ValueError(f"Model {model_name} is not supported.")
        
    return model

def unfreeze_last_k_layers(model, model_name: str, k: int):
    """
    Unfreeze the last `k` high-level blocks/layers of the model.
    """
    # Step 1: Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Step 2: Define which layers to consider
    if model_name.startswith("resnet"):
        layers = [model.layer1, model.layer2, model.layer3, model.layer4]
        if hasattr(model, "fc"):  # Unfreeze classifier
            for p in model.fc.parameters():
                p.requires_grad = True

    elif model_name.startswith("efficientnet"):
        layers = list(model.features.children())
        if hasattr(model, "classifier"):
            for p in model.classifier.parameters():
                p.requires_grad = True

    elif model_name.startswith("convnext"):
        # ConvNeXt has stem + stages + head
        layers = list(model.stem.children()) + list(model.stages.children())
        if hasattr(model, "head"):
            for p in model.head.parameters():
                p.requires_grad = True
    elif model_name.startswith("swin"):
        # For Swin Transformer, layers are stored in 'model._modules['layers']'
        layers = []
        for i, stage in enumerate(model._modules['layers']):
            # Each stage is indexed, and we append the blocks in the stage to the layers list
            layers.extend(model._modules['layers'][i].blocks)
    else:
        raise ValueError(f"Unfreezing not supported for model {model_name}")

    total_layers = len(layers)
    if k > total_layers:
        print(f"Warning: Model has only {total_layers} layers. Unfreezing all available layers.")
        k = total_layers  # Limit k to available layers
    # Step 3: Unfreeze the last `k` layers
    if k > 0:
        for layer in layers[-k:]:
            for param in layer.parameters():
                param.requires_grad = True
    
    print(f"Unfroze {k} layers for model {model_name}")



def transform_images(dataset_class):
    name = dataset_class.__name__.lower()
    if "fashion" in name:
        return 28
    elif "emotions" in name:
        return 48
    elif "flowers" in name:
        return 224
    elif "skin_cancer" in name:
        return 224
    else:
        return 128