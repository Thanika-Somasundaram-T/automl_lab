"""AutoML class for classification tasks.

Example AutoML class that trains a model on the full training set and predicts on test set.
"""

from __future__ import annotations

from typing import Any, Tuple

import torch
import numpy as np
import logging

from torch import nn, optim
from torchvision import transforms

from automl.utils import build_model, calculate_mean_std, get_data_loader, get_device

logger = logging.getLogger(__name__)

class AutoML:
    def __init__(self, seed: int) -> None:
        self.seed = seed
        self._model: nn.Module | None = None
        self._transform = None
        self.best_params = None
        self.device = get_device()
        print(f"Using device: {self.device}")

    def fit(
        self,
        dataset_class: Any,
        cfg: dict,
        epochs: int=5,
        is_val: bool=False,
    ) -> AutoML:
        
        dataset_name = dataset_class.__name__.lower()

        if "flowers" in dataset_name:
            resize_size = 128
        elif "emotions" in dataset_name:
            resize_size = 48
        elif "fashion" in dataset_name:
            resize_size = 28
        else:
            resize_size = 64  # fallback

        # Resize + convert to 3 channels for pretrained models
        self._transform = transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(*calculate_mean_std(dataset_class)),
        ])
        
        # Load full train dataset (no val split)
        train_loader, val_loader = get_data_loader(
            dataset_class=dataset_class,
            transform=self._transform,
            batch_size=cfg["batch_size"],
            split="train",
            is_val=is_val,
            seed=self.seed
        )

        # Build model and send to device
        model = build_model(cfg["model"], dataset_class.num_classes)
        model.to(self.device)
        print("model built")
        print()

        # Optimizer setup
        if cfg["optimizer"] == "adam":
            optimizer = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        else:
            optimizer = optim.SGD(model.parameters(), lr=cfg["lr"], momentum=0.9, weight_decay=cfg["weight_decay"])

        criterion = nn.CrossEntropyLoss()

        # Train loop
        model.train()
        for epoch in range(epochs):
            print("Train Epoch: ", epoch)
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()

        self._model = model
        
        if is_val:
            # validate on val_loader and return val accuracy
            acc = self.validate(val_loader)
            return acc
        else:
            return self
    
    def validate(self, val_loader):
        self._model.eval()
        correct = total = 0
        with torch.no_grad():
            print("validating...")
            for xb, yb in val_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                preds = self._model(xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        return correct / total

    def predict(self, dataset_class) -> Tuple[np.ndarray, np.ndarray]:
        # Prepare data loader for test split
        data_loader, _ = get_data_loader(dataset_class=dataset_class, transform=self._transform, batch_size=100, split="test", seed=self.seed)

        predictions = []
        labels = []

        self._model.eval()
        with torch.no_grad():
            for data, target in data_loader:
                data = data.to(self.device)
                output = self._model(data)
                predicted = torch.argmax(output, dim=1)
                labels.append(target.numpy())
                predictions.append(predicted.cpu().numpy())

        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)
        return predictions, labels
