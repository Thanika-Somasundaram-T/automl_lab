from pathlib import Path
import torch
import numpy as np
from automl.model import Network, NetworkFixed
from automl.utils import get_device
import logging
import torch.nn as nn

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(proxy_model_path: Path, model_path: Path, device, loaders):
    criterion = nn.CrossEntropyLoss()
    num_classes_dict = {d['name']: d['num_classes'] for d in loaders}

    # ---- Step 1: Load the searched proxy model ----
    proxy_model = Network(
        C=12,
        num_classes_dict=num_classes_dict,
        layers=4,
        criterion=criterion
    ).to(device)

    if proxy_model_path is not None and proxy_model_path.exists():
        logger.info(f"Loading proxy model weights from {proxy_model_path}")
        checkpoint = torch.load(proxy_model_path, map_location=device)
        state_dict = checkpoint["best_model_state"] if isinstance(checkpoint, dict) and "best_model_state" in checkpoint else checkpoint
        model_state = proxy_model.state_dict()
        filtered_state = {k: v for k, v in state_dict.items() if k in model_state and v.shape == model_state[k].shape}
        model_state.update(filtered_state)
        proxy_model.load_state_dict(model_state)
    else:
        logger.warning("Proxy model path not found. Using randomly initialized proxy model.")

    # ---- Step 2: Extract genotype from proxy model ----
    genotype = proxy_model.genotype()

    # ---- Step 3: Build the final fixed model ----
    fixed_model = NetworkFixed(
        C=32,
        num_classes_dict=num_classes_dict,
        layers=10,
        genotype=genotype,
    ).to(device)

    # ---- Step 4: Optionally load fixed model weights ----
    if model_path is not None and model_path.exists():
        logger.info(f"Loading trained fixed model weights from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
        model_state = fixed_model.state_dict()
        filtered_state = {k: v for k, v in state_dict.items() if k in model_state and v.shape == model_state[k].shape}
        model_state.update(filtered_state)
        fixed_model.load_state_dict(model_state)
    else:
        logger.warning("No trained fixed model weights found. Using freshly initialized model.")

    return fixed_model
