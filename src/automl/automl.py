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

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple, Optional
from tqdm import tqdm

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

        model = build_model(config["model"], dataset_class.num_classes)
        model.to(self.device)
        
        if config["optimizer"] == "adam":
            optimizer = optim.Adam(model.parameters(), lr=config["lr"])
        else:
            optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9)



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
    
    

    def train_one_epoch(model: nn.Module, 
                        dataloader: DataLoader, 
                        criterion: nn.Module, 
                        optimizer: torch.optim.Optimizer, 
                        device: torch.device) -> float:
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in tqdm(dataloader, desc="Training", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            total += inputs.size(0)
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def validate(model: nn.Module, 
                dataloader: DataLoader, 
                criterion: nn.Module, 
                device: torch.device) -> Tuple[float, float]:
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(dataloader, desc="Validating", leave=False):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == targets).sum().item()
                total += inputs.size(0)
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def train(model: nn.Module,
            train_loader: DataLoader,
            val_loader: Optional[DataLoader],
            criterion: nn.Module,
            optimizer: torch.optim.Optimizer,
            device: torch.device,
            num_epochs: int = 20,
            scheduler=None,
            early_stopping_patience: Optional[int] = None) -> None:
        """
        Full training loop with optional validation and early stopping.
        """
        best_val_acc = 0.0
        epochs_no_improve = 0

        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")

            train_loss, train_acc = self.train_one_epoch(model, train_loader, criterion, optimizer, device)
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

            if val_loader is not None:
                val_loss, val_acc = self.validate(model, val_loader, criterion, device)
                print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

                if scheduler is not None:
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(val_loss)
                    else:
                        scheduler.step()

                # Early stopping
                if early_stopping_patience is not None:
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        epochs_no_improve = 0
                        # Save best model weights
                        torch.save(model.state_dict(), "best_model.pth")
                    else:
                        epochs_no_improve += 1
                        if epochs_no_improve >= early_stopping_patience:
                            print("Early stopping triggered.")
                            break
            else:
                # No val_loader, just step scheduler if exists
                if scheduler is not None:
                    scheduler.step()

    # evaluate.py

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple, List, Optional

def evaluate(model: torch.nn.Module,
             dataloader: DataLoader,
             device: torch.device,
             save_preds_path: Optional[str] = None) -> Tuple[float, List[int]]:
    """
    Evaluate model on test data.

    Args:
        model: Trained PyTorch model.
        dataloader: DataLoader for test dataset.
        device: torch.device.
        save_preds_path: Optional path to save predicted labels.

    Returns:
        accuracy, list of predicted labels.
    """
    model.eval()
    correct = 0
    total = 0
    all_preds = []

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            outputs = outputs.view(-1, 10)
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            all_preds.extend(preds.cpu().tolist())

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")

    if save_preds_path is not None:
        with open(save_preds_path, "w") as f:
            for pred in all_preds:
                f.write(f"{pred}\n")

    return accuracy, all_preds




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
