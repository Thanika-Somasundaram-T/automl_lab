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
    vit_b_16, ViT_B_16_Weights,
    mobilenet_v3_large, MobileNet_V3_Large_Weights,
    efficientnet_b3, EfficientNet_B3_Weights,
    resnext50_32x4d, ResNeXt50_32X4D_Weights,
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
class SwinClassifierHead(nn.Module):
    def __init__(self, in_features, num_classes, hidden_dim=512, dropout=0.5, layers=2):
        super().__init__()
        # MLP head as before, can reuse your make_mlp_head function or define inline
        self.mlp_head = make_mlp_head(in_features, num_classes, hidden_dim, dropout, layers)
    
    def forward(self, x):
        # x shape: [B, H, W, C] e.g. [42, 7, 7, 7]
        B, H, W, C = x.shape
        x = x.view(B, H * W, C)       # flatten spatial tokens
        x = x.mean(dim=1)             # global average pooling over tokens → [B, C]
        x = self.mlp_head(x)          # [B, num_classes]
        return x

def make_mlp_head(in_features: int, num_classes: int, hidden_dim: int = 512, dropout: float = 0.5, layers: int = 2) -> nn.Sequential:
    """
    Creates an MLP head with `layers` layers, each having `hidden_dim` neurons,
    ReLU activations, dropout, and final classification layer.
    """
    layers_list = []
    current_in = in_features

    for i in range(layers - 1):
        layers_list.append(nn.Linear(current_in, hidden_dim))
        layers_list.append(nn.ReLU(inplace=True))
        layers_list.append(nn.Dropout(dropout))
        current_in = hidden_dim

    # Final output layer
    layers_list.append(nn.Linear(current_in, num_classes))
    return nn.Sequential(*layers_list)


def get_model(model_name: str, num_classes: int, hidden_dim: int = 512, dropout: float = 0.5, layers: int = 2) -> nn.Module:
    """
    Loads a pretrained model and replaces its classifier/head with a custom MLP head.
    Supports CNNs and transformers.
    """
    if model_name == 'resnet50':
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        model.fc = make_mlp_head(model.fc.in_features, num_classes, hidden_dim, dropout, layers)

    elif model_name == 'resnext50_32x4d':
        weights = models.ResNeXt50_32X4D_Weights.DEFAULT
        model = models.resnext50_32x4d(weights=weights)
        model.fc = make_mlp_head(model.fc.in_features, num_classes, hidden_dim, dropout, layers)

    elif model_name == 'efficientnet_v2_s':
        model = timm.create_model("tf_efficientnetv2_s", pretrained=True)
        model.classifier = make_mlp_head(model.classifier.in_features, num_classes, hidden_dim, dropout, layers)

    elif model_name == 'swin_tiny':
        model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True)
        in_features = model.head.in_features
        model.head = SwinClassifierHead(in_features, num_classes, hidden_dim, dropout, layers)
        
    else:
        raise ValueError(f"Model {model_name} is not supported.")

    # Freeze backbone
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze only the new head
    for name, module in model.named_children():
        if any(key in name for key in ["fc", "classifier", "head"]):
            for p in module.parameters():
                p.requires_grad = True
            
    print("Added ", layers, "to ", model_name)
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

    elif model_name == "efficientnet_v2_s":
        if hasattr(model, "blocks"):
            layers = list(model.blocks.children())
        else:
            raise ValueError(f"EfficientNet model '{model_name}' from timm does not have attribute 'blocks'.")
        
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
        if hasattr(model, "head"):
            for p in model.head.parameters():
                p.requires_grad = True
        if hasattr(model, "norm"):
            for p in model.norm.parameters():
                p.requires_grad = True
            
    elif model_name.startswith("mobilenet"):
        layers = list(model.features.children())
        for p in model.classifier.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"Unfreezing not supported for model {model_name}")

    total_layers = len(layers)
    # Step 4: Unfreeze last k blocks
    if k > total_layers:
        print(f"Warning: Model has only {total_layers} layers. Unfreezing all available layers.")
        k = total_layers  # Limit k to available layers
    # Step 3: Unfreeze the last `k` layers
    if k > 0:
        for layer in layers[-k:]:
            for param in layer.parameters():
                param.requires_grad = True
        print(f"Unfroze {k} layers for model {model_name}")
        
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError(f"No trainable parameters found for model {model_name} with k={k}.")

    return trainable_params



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
    
    
def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    """
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)

    # Use white text if squares are dark; otherwise black.
    threshold = cm_norm.max() / 2.

    for i, j in np.ndindex(cm.shape):
        color = "white" if cm_norm[i, j] > threshold else "black"
        plt.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)",
                 horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure


def print_class_distribution(dataset):
    """
    Prints class distribution for a dataset or Subset.
    Works with datasets having .targets or .labels.
    """
    if isinstance(dataset, Subset):
        labels = torch.tensor(dataset.dataset.targets)[dataset.indices]
    elif hasattr(dataset, 'targets'):
        labels = dataset.targets
    elif hasattr(dataset, 'labels'):
        labels = dataset.labels
    else:
        raise AttributeError("Dataset does not have 'targets' or 'labels' attribute")

    counts = Counter(labels.tolist())
    print("\nClass distribution:")
    for cls_idx, count in sorted(counts.items()):
        print(f"Class {cls_idx}: {count} samples")
    return counts