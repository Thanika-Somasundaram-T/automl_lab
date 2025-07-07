"""AutoML class for regression tasks.

This module contains an example AutoML class that simply returns predictions of a quickly trained MLP.
You do not need to use this setup, and you can modify this however you like.
"""
from __future__ import annotations



from typing import Any, Tuple

import torch
import random
import numpy as np
import logging

from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from automl.dummy_model import DummyNN
from automl.utils import calculate_mean_std

import optuna
from optuna.trial import Trial

from torchvision.models import resnet18, mobilenet_v2, efficientnet_b0
from torchvision.models import ResNet18_Weights, MobileNet_V2_Weights, EfficientNet_B0_Weights


logger = logging.getLogger(__name__)


class AutoML:

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self._model: nn.Module | None = None
        self._transform = None
        self.best_params = None

    def _get_data_loader(self, dataset_class, transform, batch_size, split="train"):
        dataset = dataset_class(
            root="./data",
            split=split,
            download=True,
            transform=transform
        )

        if split == "train":
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = random_split(
                dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(self.seed)
            )
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            return train_loader, val_loader
        else:
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            return data_loader, None

    def _build_model(self, model_name: str, num_classes: int) -> nn.Module:
        if model_name == "resnet18":
            weights = ResNet18_Weights.IMAGENET1K_V1
            model = resnet18(weights=weights)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
        elif model_name == "mobilenet_v2":
            weights = MobileNet_V2_Weights.IMAGENET1K_V1
            model = mobilenet_v2(weights=weights)
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, num_classes)
        elif model_name == "efficientnet_b0":
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1
            model = efficientnet_b0(weights=weights)
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, num_classes)
        else:
            raise ValueError(f"Unsupported model architecture: {model_name}")

        return model


    
    def fit(
        self,
        dataset_class: Any,
    ) -> AutoML:
        """A reference/toy implementation of a fitting function for the AutoML class.
        """
        # set seed for pytorch training
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)

        # Ensure deterministic behavior in CuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self._transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(*calculate_mean_std(dataset_class)),
            ]
        )

        def objective(trial: Trial):
            print("testing hyperpaerameters")
            # Sample hyperparameters
            model_name = trial.suggest_categorical("model", ["resnet18", "mobilenet_v2", "efficientnet_b0"])
            lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
            weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
            optimizer_name = trial.suggest_categorical("optimizer", ["adam", "sgd"])
            batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

            # Create model
            model = self._build_model(model_name, dataset_class.num_classes)

            # Data split
            train_loader, val_loader = self._get_data_loader(dataset_class, self._transform, batch_size, split="train")


            # Optimizer
            if optimizer_name == "adam":
                optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            else:
                optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

            criterion = nn.CrossEntropyLoss()

            # Train for few epochs (fidelity dimension)
            print("selected training params ==== ")
            print("model: ", model_name)
            print("batch_size: ", batch_size)
            print("lr: ", lr)
            print("weight_decay: ", weight_decay)
            print("optimizer: ", optimizer_name)
            
            model.train()
            for epoch in range(3): 
                print("Train Epoch: ", epoch)
                for xb, yb in train_loader:
                    optimizer.zero_grad()
                    out = model(xb)
                    loss = criterion(out, yb)
                    loss.backward()
                    optimizer.step()
            print()

            # Evaluate on validation
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    print("Eval ")
                    out = model(xb)
                    preds = out.argmax(dim=1)
                    correct += (preds == yb).sum().item()
                    total += yb.size(0)

            return correct / total  # top-1 accuracy
        
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=10)
        self.best_params = study.best_params

        # Train final model with best hyperparameters
        best_model_name = self.best_params["model"]
        best_batch_size = self.best_params["batch_size"]
        best_lr = self.best_params["lr"]
        best_weight_decay = self.best_params["weight_decay"]
        best_optimizer = self.best_params["optimizer"]
        print("selected best params ==== ")
        print("model: ", best_model_name)
        print("batch_size: ", best_batch_size)
        print("lr: ", best_lr)
        print("weight_decay: ", best_weight_decay)
        print("optimizer: ", best_optimizer)
    
        model = self._build_model(best_model_name, dataset_class.num_classes)
        train_loader, _ = self._get_data_loader(dataset_class, self._transform, best_batch_size, split="train")

        if best_optimizer == "adam":
            optimizer = optim.Adam(model.parameters(), lr=best_lr, weight_decay=best_weight_decay)
        else:
            optimizer = optim.SGD(model.parameters(), lr=best_lr, momentum=0.9, weight_decay=best_weight_decay)

        criterion = nn.CrossEntropyLoss()
        model.train()
        for epoch in range(5):
            print("Final train epoch, ", epoch)
            for xb, yb in train_loader:
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()

        self._model = model
        return self

    def predict(self, dataset_class) -> Tuple[np.ndarray, np.ndarray]:
        """A reference/toy implementation of a prediction function for the AutoML class.
        """
        data_loader, _ = self._get_data_loader(dataset_class, self._transform, batch_size=100, split="test")
        predictions = []
        labels = []
        self._model.eval()
        with torch.no_grad():
            for data, target in data_loader:
                output = self._model(data)
                predicted = torch.argmax(output, 1)
                labels.append(target.numpy())
                predictions.append(predicted.numpy())
        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)
        
        return predictions, labels
