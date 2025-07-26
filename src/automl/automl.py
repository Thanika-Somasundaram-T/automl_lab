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
        self.train_losses = []
        self.val_losses = []
        self.accuracy = 0
        self.device = get_device()
        print(f"Using device: {self.device}")
        
    def transform_images(self, dataset_class):
        dataset_name = dataset_class.__name__.lower()

        if "flowers" in dataset_name:
            resize_size = 128
        elif "emotions" in dataset_name:
            resize_size = 48
        elif "fashion" in dataset_name:
            resize_size = 28
        else:
            resize_size = 128  # fallback

        # Resize + convert to 3 channels for pretrained models
        self._transform = transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(*calculate_mean_std(dataset_class)),
        ])

    def fit(
        self,
        dataset_class: Any,
        config: dict,
        epochs: int=5,
        is_val: bool=False,
    ) -> AutoML:
        
        print("??????????????????", config)
        
        self.transform_images(dataset_class)
        train_loader, val_loader = get_data_loader(
            dataset_class=dataset_class,
            transform=self._transform,
            batch_size=config["batch_size"],
            split="train",
            is_val=is_val,
            seed=self.seed
        )

        model = build_model(config, dataset_class.num_classes)
        model.to(self.device)
        if config["optimizer"] == "adam":
            optimizer = optim.Adam(model.parameters(), lr=config["lr"])
        else:
            optimizer = optim.SGD(model.parameters(), lr=config["lr"])



        criterion = nn.CrossEntropyLoss()

        self.train_losses = []
        self.val_losses = []
        best_val_acc = 0.0  # Optional: to track best val accuracy

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            batch_count = 0
            print(f"Train Epoch: {epoch}")

            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                batch_count += 1

            avg_epoch_loss = epoch_loss / batch_count
            self.train_losses.append(avg_epoch_loss)
            print(f"Epoch {epoch} train loss: {avg_epoch_loss:.4f}")
            if is_val:
                # Validate after each epoch
                model.eval()
                val_loss = 0
                val_batch_count = 0
                correct = 0
                total = 0

                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb, yb = xb.to(self.device), yb.to(self.device)
                        outputs = model(xb)
                        loss = criterion(outputs, yb)
                        val_loss += loss.item()
                        val_batch_count += 1

                        preds = outputs.argmax(dim=1)
                        correct += (preds == yb).sum().item()
                        total += yb.size(0)

                avg_val_loss = val_loss / val_batch_count
                self.val_losses.append(avg_val_loss)
                val_acc = correct / total
                print(f"Epoch {epoch} val loss: {avg_val_loss:.4f}, val accuracy: {val_acc:.4f}")
                
                if val_acc > best_val_acc:
                    print("============================= best at this trial so far: ", epoch)
                    best_val_acc = val_acc
                    best_model_state = model.state_dict()
                    self.accuracy = val_acc

        # After training done, save best model if validation was used
        if is_val and best_model_state is not None:
            model.load_state_dict(best_model_state)
            self._model = model
        else:
            # No validation, just save the last model
            self._model = model
            if not is_val:
                self.accuracy = 0.0  # accuracy unknown without validation

        return self


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
                labels.append(target.cpu().numpy())
                predictions.append(predicted.cpu().numpy())

        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)
        return predictions, labels
