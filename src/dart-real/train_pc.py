import torch
import torch.nn as nn
import torch.optim as optim
import mup
from model import Network  # your mup DARTS model with alphas and betas
from datasets import get_dataloader  # assume you have a helper to get dataloaders

def train_epoch(train_loader, model, weight_optimizer, arch_optimizer, criterion, device):
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # Step 1: Update weights
        weight_optimizer.zero_grad()
        arch_optimizer.zero_grad()  # zero grads for arch too since we do two-step training

        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        weight_optimizer.step()

        # Step 2: Update architecture parameters (alphas, betas)
        # Here, normally you use validation data, but simplified here for example
        arch_optimizer.zero_grad()
        # You could calculate loss on a validation batch for arch update
        # e.g. val_inputs, val_targets = next(val_loader)
        # val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
        # val_logits = model(val_inputs)
        # arch_loss = criterion(val_logits, val_targets)
        # arch_loss.backward()
        # arch_optimizer.step()

        # For demo, skipping arch update batch here

def validate(val_loader, model, criterion, device):
    model.eval()
    total_loss, total_correct, total_samples = 0., 0, 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss = criterion(logits, targets)
            total_loss += loss.item() * inputs.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == targets).sum().item()
            total_samples += inputs.size(0)
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyperparams (tune these via NePS)
    weight_lr = 1e-3       # MuP-scaled learning rate for weights
    arch_lr = 3e-4         # Smaller LR for architecture parameters (alphas, betas)
    weight_wd = 1e-4       # Weight decay for weights
    arch_wd = 0.0          # Usually zero for architecture params

    criterion = nn.CrossEntropyLoss()

    # Load datasets
    train_loader = get_dataloader('train', batch_size=64)
    val_loader = get_dataloader('val', batch_size=64)
    test_loader = get_dataloader('test', batch_size=64)

    # Create model
    model = Network(C=16, num_classes=10, layers=8, criterion=criterion)
    model.to(device)

    # Initialize MuP on model
    mup.set_base_shapes(model)  # important before creating MuAdamW optimizer

    # Separate params
    weight_params = []
    arch_params = []
    for name, param in model.named_parameters():
        if 'alphas' in name or 'betas' in name:
            arch_params.append(param)
        else:
            weight_params.append(param)

    # Optimizers
    weight_optimizer = mup.MuAdamW(weight_params, lr=weight_lr, weight_decay=weight_wd)
    arch_optimizer = optim.Adam(arch_params, lr=arch_lr, weight_decay=arch_wd)

    epochs = 40

    for epoch in range(epochs):
        train_epoch(train_loader, model, weight_optimizer, arch_optimizer, criterion, device)

        val_loss, val_acc = validate(val_loader, model, criterion, device)
        print(f"Epoch {epoch+1}/{epochs} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # Test eval
    test_loss, test_acc = validate(test_loader, model, criterion, device)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

if __name__ == '__main__':
    main()
