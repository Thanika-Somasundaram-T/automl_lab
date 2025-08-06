from pathlib import Path
import time
from typing import Tuple
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from automl.utils import get_data_loader, get_device
from automl.model import Network, NetworkFixed
import io
from PIL import Image
from torchvision.transforms import ToTensor

device = get_device()
print("Using device:", device)

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
    

def figure_to_tensor(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    return ToTensor()(img)

def train_one_epoch(model, optimizer_w, dataloaders, dataset_names, criterion, grad_clip, max_batches_per_dataset=200):
    """
    Train one epoch across all datasets (multitask).
    """
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0

    all_batches = []
    for i, dl in enumerate(dataloaders):
        for j, batch in enumerate(dl):
            # if j >= max_batches_per_dataset:
            #     break
            all_batches.append((i, batch))

    progress_bar = tqdm(all_batches, desc="Training", leave=False)

    for i, (inputs, targets) in progress_bar:
        inputs, targets = inputs.to(device), targets.to(device)

        logits = model(inputs, dataset_name=dataset_names[i])
        loss = criterion(logits, targets)

        optimizer_w.zero_grad()
        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer_w.step()

        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        _, preds = torch.max(logits, 1)
        total_correct += (preds == targets).sum().item()
        total_samples += batch_size

        progress_bar.set_postfix({
            "loss": f"{total_loss / total_samples:.4f}",
            "acc": f"{total_correct / total_samples:.4f}",
        })

    return total_loss / total_samples, total_correct / total_samples


def evaluate(model, dataloaders, dataset_names, criterion, max_batches_per_dataset=200):
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
            # if j >= max_batches_per_dataset:
            #     break
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

    return total_loss / total_samples, total_correct / total_samples, all_preds, all_targets


def get_optimizer(fixed_model, lr_w, weight_decay_w):
    return torch.optim.Adam(fixed_model.parameters(), lr=lr_w, weight_decay=weight_decay_w)


def fit3(config, loaders, best_model_path, trial_name, seed=42):
    """
    Train final fixed architecture with early stopping.
    """
    print("fit 3")  
    trial_dir = Path("./tensorboard/final") / trial_name
    writer = SummaryWriter(log_dir=trial_dir)

    train_loaders = [d['train_loader'] for d in loaders]
    val_loaders = [d['val_loader'] for d in loaders]
    dataset_names = [d['name'] for d in loaders]
    num_classes_dict = {d['name']: d['num_classes'] for d in loaders}

    # Hyperparams
    lr_w = config["lr_w"]
    weight_decay = config["weight_decay"]
    grad_clip = config["grad_clip"]
    max_epochs = config["max_epochs"]
    patience = config.get("early_stopping_patience", 10)

    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0
    best_model_state = None
    epochs_no_improve = 0

    # Load searched proxy model
    proxy_model = Network(
        C=config.get("base_width", 12),
        num_classes_dict=num_classes_dict,
        layers=config.get("layers", 4),
        criterion=criterion
    ).to(device)

    if best_model_path is not None and best_model_path.exists():
        print(f"Loading pretrained weights from {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=device)
        state_dict = checkpoint["best_model_state"] if isinstance(checkpoint, dict) and "best_model_state" in checkpoint else checkpoint
        model_state = proxy_model.state_dict()
        filtered_state = {k: v for k, v in state_dict.items() if k in model_state and v.shape == model_state[k].shape}
        model_state.update(filtered_state)
        proxy_model.load_state_dict(model_state)
        
    genotype = proxy_model.genotype()

    # Build fixed model from genotype
    fixed_model = NetworkFixed(
        C=config.get("final_width", 32),
        num_classes_dict=num_classes_dict,
        layers=config.get("final_layers", 15),
        genotype=genotype,
    ).to(device)

    optimizer_w = get_optimizer(fixed_model, lr_w, weight_decay)

    # Main loop with early stopping
    epoch_bar = tqdm(range(max_epochs), desc="Epochs")
    for epoch in epoch_bar:
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            fixed_model, optimizer_w,
            train_loaders, dataset_names, criterion, grad_clip
        )
        val_loss, val_acc, all_preds, all_targets = evaluate(fixed_model, val_loaders, dataset_names, criterion)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        writer.add_scalar("LearningRate/weights", lr_w, epoch)

         # Track best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = fixed_model.state_dict()
            epochs_no_improve = 0
            for dataset_name in dataset_names:
                num_cls = num_classes_dict[dataset_name]
                class_names = [str(i) for i in range(num_cls)]
                log_per_class_accuracy(writer, all_preds[dataset_name], all_targets[dataset_name], class_names, dataset_name, epoch)
        else:
            epochs_no_improve += 1

        # Check patience **after** updating counter
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch + 1} (no improvement for {patience} epochs)")
            break

        epoch_time = time.time() - start_time
        epoch_bar.set_postfix({
            "train_loss": f"{train_loss:.4f}",
            "val_acc": f"{val_acc:.4f}",
            "epoch_time": f"{epoch_time:.1f}s",
        })

    writer.close()

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    
    if best_model_state is not None:
        fixed_model.load_state_dict(best_model_state)
        save_path = Path("./saved_models") / f"{trial_name}_best_model.pth"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_model_state, save_path)
        print(f"Saved best model to {save_path}")

    return {
        "val_acc": best_val_acc,
        "model": fixed_model,
        "best_config": config
    }