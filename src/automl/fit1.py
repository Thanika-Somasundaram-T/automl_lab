import datetime
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
import numpy as np
import time

from automl.model import Network
from . import utils
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image
from torchvision.transforms import ToTensor
import itertools


device = utils.get_device()
print("taken device: ", device)

def log_per_class_accuracy(writer, preds, targets, class_names, dataset_name, epoch):
    preds = np.array(preds)
    targets = np.array(targets)
    num_classes = len(class_names)
    per_class_acc = []
    for cls in range(num_classes):
        cls_mask = (targets == cls)
        if cls_mask.sum() == 0:
            acc = 0.0
        else:
            acc = (preds[cls_mask] == targets[cls_mask]).mean()
        per_class_acc.append(acc)

    # Plot as a bar chart
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(num_classes), per_class_acc)
    ax.set_xticks(range(num_classes))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_ylim(0, 1)
    ax.set_title(f"Per-Class Accuracy ({dataset_name})")
    ax.set_ylabel("Accuracy")

    writer.add_image(f"PerClassAccuracy/{dataset_name}", figure_to_tensor(fig), epoch)
    plt.close(fig)


def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(class_names)),
           yticks=np.arange(len(class_names)),
           xticklabels=class_names,
           yticklabels=class_names,
           ylabel='True label',
           xlabel='Predicted label',
           title='Confusion Matrix')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig

def figure_to_tensor(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    return ToTensor()(img)


def get_optimizer(model, lr_w, lr_alpha, weight_decay_w):
    """
    Separate model parameters into:
      - weight_params: standard network weights (conv, bn, fc)
      - arch_params: architecture parameters (alphas, betas)
    Returns two optimizers: Adam for weights, Adam for alphas.
    """
    weight_params = []
    arch_params = []
    for name, param in model.named_parameters():
        if 'alphas' in name or 'betas' in name:
            arch_params.append(param)
        else:
            weight_params.append(param)

    optimizer_w = torch.optim.Adam(weight_params, lr=lr_w, weight_decay=weight_decay_w)
    optimizer_alpha = torch.optim.Adam(arch_params, lr=lr_alpha, weight_decay=0)
    return optimizer_w, optimizer_alpha

def train_one_epoch(model, optimizer_w, optimizer_alpha, dataloaders, dataset_names, criterion, grad_clip, max_batches_per_dataset=60):
    """
    Train one epoch across all datasets (multitask):
      - Each dataset's batches are interleaved
      - We only use up to `max_batches_per_dataset` batches per dataset to speed up proxy training
    """
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0

    # Build a mixed list of batches (dataset index + data)
    all_batches = []
    for i, dl in enumerate(dataloaders):
        for j, batch in enumerate(dl):
            if j >= max_batches_per_dataset:  # limit for proxy mode
                break
            all_batches.append((i, batch))

    progress_bar = tqdm(all_batches, desc="Training", leave=False)

    for i, (inputs, targets) in progress_bar:
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        logits = model(inputs, dataset_name=dataset_names[i])
        loss = criterion(logits, targets)

        # Zero grads for both optimizers
        optimizer_w.zero_grad()
        optimizer_alpha.zero_grad()

        # Backward pass
        loss.backward()

        # Clip only weight gradients (NOT architecture params)
        if grad_clip > 0:
            # Clip only non-architecture parameters
            torch.nn.utils.clip_grad_norm_(
                [p for name, p in model.named_parameters() 
                if 'alphas' not in name and 'betas' not in name],
                grad_clip
    )

        # Optimizer steps
        optimizer_w.step()
        optimizer_alpha.step()

        # Metrics
        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        _, preds = torch.max(logits, 1)
        total_correct += (preds == targets).sum().item()
        total_samples += batch_size

        progress_bar.set_postfix({
            "loss": f"{total_loss / total_samples:.4f}",
            "acc": f"{total_correct / total_samples:.4f}",
        })

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc

def evaluate(model, dataloaders, dataset_names, criterion, max_batches_per_dataset=60):
    """
    Evaluate on a subset of validation data for speed.
    """
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    
    all_preds = {name: [] for name in dataset_names}
    all_targets = {name: [] for name in dataset_names}

    all_batches = []
    for i, dl in enumerate(dataloaders):
        for j, batch in enumerate(dl):
            if j >= max_batches_per_dataset:  # proxy mode eval
                break
            all_batches.append((i, batch))

    progress_bar = tqdm(all_batches, desc="Validating", leave=False)

    with torch.no_grad():
        for i, (inputs, targets) in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs, dataset_name=dataset_names[i])
            loss = criterion(logits, targets)

            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            _, preds = torch.max(logits, 1)
            total_correct += (preds == targets).sum().item()
            total_samples += batch_size
            
            all_preds[dataset_names[i]].extend(preds.cpu().numpy())
            all_targets[dataset_names[i]].extend(targets.cpu().numpy())


            progress_bar.set_postfix({
                "val_loss": f"{total_loss / total_samples:.4f}",
                "val_acc": f"{total_correct / total_samples:.4f}",
            })
            

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    return avg_loss, avg_acc, all_preds, all_targets

def fit1(config, loaders, trial_name, seed=42):
    """
    Proxy training function:
      - Trains a smaller PC-DARTS network on a subset of data
      - Optimizes both network weights and architecture params
      - Returns best validation accuracy & model state
    """
    print("fit 1")
    trial_dir = Path("./tensorboard") / trial_name
    writer = SummaryWriter(log_dir=trial_dir)
    
    train_loaders = [d['train_loader'] for d in loaders]
    val_loaders = [d['val_loader'] for d in loaders]
    dataset_names = [d['name'] for d in loaders]
    num_classes_dict = {d['name']: d['num_classes'] for d in loaders}

    # Print a sample batch shape
    for i, dl in enumerate(train_loaders):
        inputs, targets = next(iter(dl))
        print(f"Train loader {dataset_names[i]} batch shape: {inputs.shape}")

    # Hyperparams
    lr_w = config["lr_w"]
    lr_alpha = config["lr_alpha"]
    weight_decay = config["weight_decay"]
    grad_clip = config["grad_clip"]
    max_epochs = config["max_epochs"]

    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0
    best_model_state = None
    best_config = None

    base_width = config.get("base_width", 12)  # smaller width for proxy
    proxy_layers = config.get("layers", 4)    # fewer layers for proxy

    model = Network(
        C=base_width, num_classes_dict=num_classes_dict,
        layers=proxy_layers, criterion=criterion
    ).to(device)

    optimizer_w, optimizer_alpha = get_optimizer(model, lr_w, lr_alpha, weight_decay)

    # Main loop
    epoch_bar = tqdm(range(max_epochs), desc="Epochs")
    for epoch in epoch_bar:
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model, optimizer_w, optimizer_alpha,
            train_loaders, dataset_names, criterion, grad_clip
        )
        val_loss, val_acc, all_targets,all_preds = evaluate(model, val_loaders, dataset_names, criterion)
        
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        writer.add_scalar("LearningRate/weights", lr_w, epoch)
        writer.add_scalar("LearningRate/alphas", lr_alpha, epoch)

        # Track best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            best_config = {
                "lr_w": lr_w,
                "lr_alpha": lr_alpha,
                "weight_decay": weight_decay,
                "grad_clip": grad_clip
            }
            for dataset_name in dataset_names:
                num_cls = num_classes_dict[dataset_name]
                class_names = [str(i) for i in range(num_cls)]
                log_per_class_accuracy(writer, all_preds[dataset_name], all_targets[dataset_name], class_names, dataset_name, epoch)

        epoch_time = time.time() - start_time
        epoch_bar.set_postfix({
            "train_loss": f"{train_loss:.4f}",
            "val_acc": f"{val_acc:.4f}",
            "epoch_time": f"{epoch_time:.1f}s",
        })
        
    writer.close()

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    print("Best hyperparameters:")
    for k, v in best_config.items():
        print(f"{k}: {v}")

    return {
        "val_acc": best_val_acc,
        "best_model_state": best_model_state,
        "best_config": best_config
    }
