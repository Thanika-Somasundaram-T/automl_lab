import torch
from torch.utils.data import DataLoader
import mup
from model import Network
from dataset_skin_cancer import SkinCancerDataset
from train import train_one_epoch, validate

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    train_dataset = SkinCancerDataset(split='train')
    val_dataset = SkinCancerDataset(split='val')

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

    criterion = torch.nn.CrossEntropyLoss()

    # Full DARTS model config
    C = 36  # normal width
    layers = 20  # full depth
    num_classes = train_dataset.num_classes

    model = Network(C=C, num_classes=num_classes, layers=layers, criterion=criterion).to(device)

    # Load tuned hyperparams from Phase 2 results
    best_lr_alpha = 1e-3  # replace with actual best
    weight_decay = 5e-4   # replace with actual best
    grad_clip = 1.0       # replace with actual best

    # Setup optimizers with mup
    weight_params = [p for n, p in model.named_parameters() if 'alphas' not in n]
    arch_params = [model.alphas_normal, model.alphas_reduce]

    weight_opt = mup.MuAdam(weight_params, lr=0.01, weight_decay=weight_decay)
    arch_opt = mup.MuAdam(arch_params, lr=best_lr_alpha, weight_decay=0)

    epochs = 60

    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            weight_opt.zero_grad()
            arch_opt.zero_grad()

            loss = model.loss(inputs, targets)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            weight_opt.step()
            arch_opt.step()

        val_acc = validate(model, val_loader, device)
        print(f"Epoch {epoch+1}/{epochs} Validation Accuracy: {val_acc:.4f}")

    # Save final model and architecture weights as needed
    torch.save(model.state_dict(), 'final_darts_skin_cancer.pth')

if __name__ == '__main__':
    main()
