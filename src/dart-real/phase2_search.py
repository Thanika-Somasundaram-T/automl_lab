import torch
from torch.utils.data import DataLoader
import mup
import random
from model import Network
from dataset_skin_cancer import SkinCancerDataset  # You must implement this or adapt a loader
from train import train_one_epoch, validate  # From your previous training script
import numpy as np

# Example: phase 1 best hyperparams (usually you'd load this from a file or config)
phase1_best = {
    'lr_alpha': 1e-3,
    'weight_decay': 5e-4,
    'grad_clip': 1.0,
}

# Build the narrower search space ±50%
def build_search_space(best_params):
    def uniform_around(val):
        return (val * 0.5, val * 1.5)
    return {
        'lr_alpha': uniform_around(best_params['lr_alpha']),
        'weight_decay': uniform_around(best_params['weight_decay']),
        'grad_clip': uniform_around(best_params['grad_clip']),
    }

search_space = build_search_space(phase1_best)

# Sampling function in the narrower range
def sample_hyperparams(space):
    return {
        'lr_alpha': 10 ** random.uniform(np.log10(space['lr_alpha'][0]), np.log10(space['lr_alpha'][1])),
        'weight_decay': 10 ** random.uniform(np.log10(space['weight_decay'][0]), np.log10(space['weight_decay'][1])),
        'grad_clip': random.uniform(space['grad_clip'][0], space['grad_clip'][1]),
    }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load skin cancer dataset splits
    train_dataset = SkinCancerDataset(split='train')
    val_dataset = SkinCancerDataset(split='val')

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

    criterion = torch.nn.CrossEntropyLoss()

    layers = 8  # proxy model depth same as phase 1
    C = 16  # proxy model width

    num_trials = 6
    epochs = 35

    best_val_acc = 0.0
    best_hparams = None

    for trial in range(num_trials):
        print(f"Trial {trial+1}/{num_trials}")

        # Sample hyperparameters from narrow search space
        hparams = sample_hyperparams(search_space)
        print(f"Hyperparams: lr_alpha={hparams['lr_alpha']:.5f}, weight_decay={hparams['weight_decay']:.5f}, grad_clip={hparams['grad_clip']:.3f}")

        # Initialize model
        model = Network(C=C, num_classes=train_dataset.num_classes, layers=layers, criterion=criterion).to(device)

        # Setup optimizer for weights and architecture params with mup optimizers
        weight_params = [p for n, p in model.named_parameters() if 'alphas' not in n]
        arch_params = [model.alphas_normal, model.alphas_reduce]

        weight_opt = mup.MuAdam(weight_params, lr=0.01, weight_decay=hparams['weight_decay'])
        arch_opt = mup.MuAdam(arch_params, lr=hparams['lr_alpha'], weight_decay=0)

        for epoch in range(epochs):
            model.train()
            for batch in train_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)

                # Zero grads
                weight_opt.zero_grad()
                arch_opt.zero_grad()

                # Forward + loss
                loss = model.loss(inputs, targets)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), hparams['grad_clip'])

                # Step optimizers
                weight_opt.step()
                arch_opt.step()

            val_acc = validate(model, val_loader, device)
            print(f"Epoch {epoch+1}/{epochs} - Val Acc: {val_acc:.4f}")

        # Save best hyperparams by val accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_hparams = hparams

    print(f"Best val acc: {best_val_acc:.4f}")
    print(f"Best hyperparams found: {best_hparams}")

if __name__ == '__main__':
    main()
