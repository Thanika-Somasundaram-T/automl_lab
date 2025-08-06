import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    # For deterministic behavior on CuDNN backend
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def transform_images(dataset_class):
    name = dataset_class.__name__.lower()
    if "fashion" in name:
        return 28
    elif "emotions" in name:
        return 48
    elif "flowers" in name:
        return 96
    elif "skin_cancer" in name:
        return 72
    else:
        return 64
    
def get_device():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def calculate_mean_std(dataset_class):
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


def get_data_loader(
    dataset_class,
    batch_size,
    split="train",
    is_val=False,
    seed=42,
    train_transform=None,
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