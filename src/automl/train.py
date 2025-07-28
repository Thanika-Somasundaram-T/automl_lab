import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR

from automl.utils import calculate_mean_std, get_data_loader, get_device, get_model, plot_confusion_matrix, set_global_seed, transform_images, unfreeze_last_k_layers
import os

def train_and_validate(
    config: dict,
    dataset_class,
    seed: int = 42
) -> float:
    """
    Train and validate the model with given config, using your loader and seed functions.

    Args:
        dataset_class: Dataset class that supports root, split, download, transform args
        config: Dict containing keys:
            - model: str, model name
            - lr: float
            - optimizer: str, 'adam' or 'sgd'
            - batch_size: int
            - max_epochs: int
            - unfreeze_layers: int
        seed: random seed for reproducibility

    Returns:
        Validation accuracy (float) after training.
    """

    set_global_seed(seed)
    device = get_device()

    # Calculate mean and std for normalization
    mean, std = calculate_mean_std(dataset_class)
    resize_size = transform_images(dataset_class)

    # Define transforms with normalization
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomResizedCrop(224), 
        # random crop + resize for augmentation
        transforms.RandomHorizontalFlip(),   # random flip augmentation
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    val_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(244),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


    # Load train and val loaders
    train_loader, val_loader = get_data_loader(
        dataset_class,
        train_transform=train_transform,
        val_transform=val_transform,
        batch_size=config["batch_size"],
        split="train",
        is_val=True,
        seed=seed
    )

    # Get test loader if needed (not used here)
    # test_loader, _ = get_data_loader(dataset_class, transform=val_transform, batch_size=config["batch_size"], split="test")

    num_classes = dataset_class.num_classes

    # Create model and move to device
    model = get_model(config["model"], num_classes=num_classes)
    model.to(device)

    # Freeze/unfreeze layers as per config
    unfreeze_last_k_layers(model, config["model"], config["unfreeze_layers"])

    # Optimizer
    params_to_optimize = filter(lambda p: p.requires_grad, model.parameters())
    if config["optimizer"] == "adam":
        optimizer = optim.Adam(params_to_optimize, lr=config["lr"])
    elif config["optimizer"] == "sgd":
        optimizer = optim.SGD(params_to_optimize, lr=config["lr"], momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {config['optimizer']}")
    
    dataset_name = dataset_class.__name__
    model_folder = config["model"]
    freeze_folder = f"unfreeze{config['unfreeze_layers']}"
    trial_name = f"{config['model']}_lr{config['lr']:.5f}_bs{config['batch_size']}"
    log_dir = os.path.join("tensor_logs", dataset_name, model_folder, freeze_folder, trial_name)
    writer = SummaryWriter(log_dir=log_dir)

    
    criterion = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)


    best_val_acc = 0.0
    global_step = 0

    for epoch in range(config["max_epochs"]):
        start_time = time.time()
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            writer.add_scalar("Loss/Train_batch", loss.item(), global_step)
            global_step += 1

        avg_train_loss = train_loss / len(train_loader.dataset)
        train_acc = accuracy_score(train_labels, train_preds)

        model.eval()
        val_preds = []
        val_labels = []
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        avg_val_loss = val_loss / len(val_loader.dataset)
        val_acc = accuracy_score(val_labels, val_preds)
        
        val_labels_np = np.array(val_labels)
        val_preds_np = np.array(val_preds)

        per_class_acc = []
        for cls in range(num_classes):
            cls_mask = val_labels_np == cls
            cls_correct = np.sum(val_preds_np[cls_mask] == cls)
            cls_total = np.sum(cls_mask)
            acc_cls = cls_correct / cls_total if cls_total > 0 else 0
            per_class_acc.append(acc_cls)

        # Precision, Recall, F1-score (macro)
        precision, recall, f1, _ = precision_recall_fscore_support(val_labels_np, val_preds_np, average='macro', zero_division=0)

        # Confusion Matrix
        cm = confusion_matrix(val_labels_np, val_preds_np)
        
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Loss/Val", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/Train_Top1", train_acc, epoch)
        writer.add_scalar("Accuracy/Val_Top1", val_acc, epoch)
        writer.add_scalar("Precision/Val", precision, epoch)
        writer.add_scalar("Recall/Val", recall, epoch)
        writer.add_scalar("F1/Val", f1, epoch)
        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        writer.add_scalar("LearningRate", current_lr, epoch)
        fig = plot_confusion_matrix(cm, list(range(num_classes)))
        writer.add_figure("Confusion_Matrix", fig, epoch)

        # Training time per epoch
        epoch_time = time.time() - start_time
        writer.add_scalar("Time/TrainEpoch", epoch_time, epoch)

        print(f"Epoch {epoch+1}/{config['max_epochs']} - "
              f"Train Loss: {avg_train_loss:.4f} - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f} - "
              f"Precision: {precision:.4f} - Recall: {recall:.4f} - F1: {f1:.4f} - "
              f"Epoch Time: {epoch_time:.1f}s")

        print(f"Epoch {epoch+1}/{config['max_epochs']} - Train Loss: {avg_train_loss:.4f} - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
    
    writer.close()
    return best_val_acc
