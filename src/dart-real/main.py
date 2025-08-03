from __future__ import annotations
import os
from pathlib import Path

from torchvision import transforms
import yaml


script_dir = Path(__file__).parent
from pathlib import Path
from sklearn.metrics import accuracy_score
import argparse

import logging
from torch.utils.tensorboard import SummaryWriter

from . import utils
from . import evaluate_pipeline


from automl.datasets import FashionDataset, FlowersDataset, EmotionsDataset, SkinCancerDataset
from neps import run

logger = logging.getLogger(__name__)


def main(
    dataset: str,
    output_path: Path,
    neps_dir: Path,
    seed: int,
    use_neps: bool,
):
    
    p1_dataset = [FashionDataset, FlowersDataset, EmotionsDataset]
    p2_dataset = SkinCancerDataset
    loaders=[]
    
    
    for dataset in p1_dataset:
                 
        mean, std = utils.calculate_mean_std(dataset)
        resize_size = 64
        
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
        
        train_loader, val_loader = utils.get_data_loader(
            dataset,
            train_transform=train_transform,
            val_transform=val_transform,
            batch_size=18,
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
        
        
    with open("./phase1_config.yaml", "r") as f:
        neps_config = yaml.safe_load(f)
        
    
    
    neps_config["evaluate_pipeline"] = evaluate_pipeline.neps_training_wrapper(loaders, seed, neps_dir=neps_dir)
    
    run(
        optimizer=neps_config["optimizer"],
        max_evaluations_total=neps_config["max_evaluations_total"],
        root_directory=neps_dir,
        pipeline_space=neps_config["pipeline_space"],
        evaluate_pipeline=neps_config["evaluate_pipeline"],
    )

 




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
        "--neps_dir",
        type=Path,
        default=Path("neps_results"),
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
        dataset=args.dataset,
        output_path=args.output_path,
        seed=args.seed,
        use_neps=args.neps,
        neps_dir=args.neps_dir
    )