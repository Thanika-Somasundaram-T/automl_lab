from typing import Tuple
import numpy as np
import torch
from tqdm import tqdm
from torchvision import transforms

from automl.utils import calculate_mean_std, get_data_loader

def predict(model: torch.nn.Module, dataset_class, device, seed) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict on the test split of the dataset_class using the given model.
    """
    
    transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(*calculate_mean_std(dataset_class)),
            ]
        )
    
    data_loader, _ = get_data_loader(
        dataset_class=dataset_class,
        test_transform=transform,
        batch_size=100,
        split="test",
        seed=seed
    )

    predictions = []
    labels = []

    model.eval()
    with torch.no_grad():
        for data, target in tqdm(data_loader, desc="Predicting", leave=False):
            data = data.to(device)
            output = model(data, dataset_class.__name__)
            predicted = torch.argmax(output, dim=1)
            labels.append(target.cpu().numpy())
            predictions.append(predicted.cpu().numpy())

    predictions = np.concatenate(predictions)
    labels = np.concatenate(labels)
    return predictions, labels