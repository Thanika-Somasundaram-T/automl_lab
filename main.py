from __future__ import annotations
import json
import os
from pathlib import Path

import numpy as np
from torchvision import transforms
import yaml


script_dir = Path(__file__).parent
from pathlib import Path
from sklearn.metrics import accuracy_score
import argparse

import logging
from torch.utils.tensorboard import SummaryWriter

from automl.utils import calculate_mean_std, get_data_loader, get_device
from automl.pipeline1 import neps_phase1_wrapper
from automl.pipeline2 import neps_phase2_wrapper
from automl.fit3 import fit3
from automl.predict import predict

from automl.datasets import FashionDataset, FlowersDataset, EmotionsDataset, SkinCancerDataset
from neps import run

logger = logging.getLogger(__name__)


def load_data(datasets_list, seed, resize_size:64, batch_size:18):
    loaders=[]
    for dataset in datasets_list:
                 
        mean, std = calculate_mean_std(dataset)

        
        train_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomResizedCrop(resize_size), 
            # random crop + resize for augmentation
            transforms.RandomHorizontalFlip(),   # random flip augmentation
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        val_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(resize_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        
        train_loader, val_loader = get_data_loader(
            dataset,
            train_transform=train_transform,
            val_transform=val_transform,
            batch_size=batch_size,
            split="train",
            is_val=True,
            seed=seed
        )
        
        loaders.append({
            "name": dataset.__name__,
            "num_classes": dataset.num_classes,
            "train_loader": train_loader,
            "val_loader": val_loader,
        })
    return loaders
    

def main(
    output_path: Path,
    search_dir: Path,
    finetune_dir: Path,
    seed: int,
):
    
    p1_dataset = [FlowersDataset, FashionDataset, EmotionsDataset]
    p2_dataset = [FashionDataset]
    
    
    
    
    # search_datasets = load_data(p1_dataset, seed, 64, 18)
        
    # with open("./phase1_config.yaml", "r") as f:
    #     neps_config = yaml.safe_load(f)
        
    # neps_config["evaluate_pipeline"] = neps_phase1_wrapper(search_datasets, seed, neps_dir=search_dir)
    
    # run(
    #     optimizer=neps_config["optimizer"],
    #     max_evaluations_total=neps_config["max_evaluations_total"],
    #     root_directory=search_dir,
    #     pipeline_space=neps_config["pipeline_space"],
    #     evaluate_pipeline=neps_config["evaluate_pipeline"],
    # )

    best_config_path = search_dir/"best_config.json"
    with open(best_config_path, "r") as f:
        best_config = json.load(f)

    with open("./phase2_config.yaml", "r") as f:
        phase2_config = yaml.safe_load(f)

    best_config.update(phase2_config)
    print("best_config", best_config)

    pipeline_space = {
        "lr_alpha": {
            "lower": best_config["lr_alpha"] * 0.8,
            "upper": best_config["lr_alpha"] * 1.2
        },
        "lr_w": {
            "lower": best_config["lr_w"] * 0.8,
            "upper": best_config["lr_w"] * 1.2
        },
        "grad_clip": {
            "lower": best_config["grad_clip"] * 0.8,
            "upper": best_config["grad_clip"] * 1.2
        },
        "weight_decay": {
            "lower": best_config["weight_decay"] * 0.8,
            "upper": best_config["weight_decay"] * 1.2
        },
        "max_epochs": {
            "lower": best_config["max_epochs"] - 5,
            "upper": best_config["max_epochs"],
            "is_fidelity": True
        }
    }
    
    finetune_dataset = load_data(p2_dataset, seed, 64, 18)
    best_config["evaluate_pipeline"] = neps_phase2_wrapper(loaders=finetune_dataset, seed=seed, neps_dir=finetune_dir, search_dir=search_dir)
    run(
        optimizer=best_config["optimizer"],
        max_evaluations_total=best_config["max_evaluations_total"],
        root_directory=finetune_dir,
        pipeline_space=pipeline_space,
        evaluate_pipeline=best_config["evaluate_pipeline"],
    )
    
    best_config_path = finetune_dir/"best_config.json"
    best_model_path = Path(finetune_dir/"best_model.pth")
    trial_name = "trial_final"
    
    with open(best_config_path, "r") as f:
        final_best_config = json.load(f)
    with open("./phase3_config.yaml", "r") as f:
        phase3_config = yaml.safe_load(f)

    final_best_config.update(phase3_config)
    
    final_dataset = load_data(p2_dataset, seed, 72, 20)
    results = fit3(config=final_best_config, loaders=final_dataset, best_model_path=best_model_path, trial_name=trial_name, seed=seed)
    trained_model = results["model"]
    dataset_ = p2_dataset[0]

    test_preds, test_labels = predict(trained_model, dataset_class=dataset_, device=get_device(), seed=seed)

    
    # Get predictions from best models
    # Write predictions to disk
    logger.info("Writing predictions to disk")
    with output_path.open("wb") as f:
        np.save(f, test_preds)

    # In case of running on the exam dataset
            
    if p2_dataset[0].__name__ == "SkinCancerDataset":
        test_output_path = Path("data/exam_dataset/predictions.npy")
        test_output_path.parent.mkdir(parents=True, exist_ok=True)
        with test_output_path.open("wb") as f:
            np.save(f, test_preds)

    # If labels are available, evaluate
    if not np.isnan(test_labels).any():
        acc = accuracy_score(test_labels, test_preds)
        logger.info(f"Accuracy on test set: {acc}")
    else:
        logger.info(f"No test split for dataset '{dataset_.__name__}'")


    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        required=False,
        help="The name of the dataset to run on.",
        choices=["fashion", "flowers", "emotions", "skin_cancer"]
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("predictions.npy"),
        help=(
            "The path to save the predictions to."
            " By default this will just save to './predictions.npy'."
        )
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Random seed for reproducibility if you are using and randomness,"
            " i.e. torch, numpy, pandas, sklearn, etc."
        )
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Whether to log only warnings and errors."
    )
    
    parser.add_argument(
        "--neps",
        action="store_true",
        help="Whether to use NEPS or not."
    )
    
    parser.add_argument(
        "--search_dir",
        type=Path,
        default=Path("search_result"),
        help=(
            "The path to save the neps_results to. "
            "By default this will save to './neps_results'."
        )
    )
    
    parser.add_argument(
        "--finetune_dir",
        type=Path,
        default=Path("finetune_result"),
        help=(
            "The path to save the neps_results to. "
            "By default this will save to './neps_results'."
        )
    )

    args = parser.parse_args()

    if not args.quiet:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    logger.info(
        f"Running dataset {args.dataset}"
        f"\n{args}"
    )

    main(
        output_path=args.output_path,
        seed=args.seed,
        search_dir=args.search_dir,
        finetune_dir=args.finetune_dir
    )