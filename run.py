"""An example run file which trains a dummy AutoML system on the training split of a dataset
and logs the accuracy score on the test set.

In the example data you are given access to the labels of the test split, however
in the test dataset we will provide later, you will not have access
to this and you will need to output your predictions for the images of the test set
to a file, which we will grade using github classrooms!
"""
from __future__ import annotations
import os
from pathlib import Path

script_dir = Path(__file__).parent
import json
from pathlib import Path
import neps
from sklearn.metrics import accuracy_score
import numpy as np
import yaml
from automl.automl import AutoML
import argparse

import logging

from automl.datasets import FashionDataset, FlowersDataset, EmotionsDataset
from neps import run
from automl.pipeline import BEST_RESULT_PATH, neps_training_wrapper
from automl.search_space import search_space
from automl.utils import plot_neps, set_global_seed

logger = logging.getLogger(__name__)


def main(
    dataset: str,
    output_path: Path,
    seed: int,
    use_neps: bool
):
    match dataset:
        case "fashion":
            dataset_class = FashionDataset
        case "flowers":
            dataset_class = FlowersDataset
        case "emotions":
            dataset_class = EmotionsDataset
        case _:
            raise ValueError(f"Invalid dataset: {args.dataset}")

    logger.info("Fitting AutoML")

    # You do not need to follow this setup or API it's merely here to provide
    # an example of how your automl system could be used.
    # As a general rule of thumb, you should **never** pass in any
    # test data to your AutoML sol
    # ution other than to generate predictions.
    set_global_seed(seed)

    best_params = None
    if use_neps:
        with open("./neps.yaml", "r") as f:
            neps_config = yaml.safe_load(f)

        neps_config["evaluate_pipeline"] = neps_training_wrapper(dataset_class, seed)

        # # Unpack dictionary directly
        run(**neps_config)
        plot_neps()
        
    if Path(BEST_RESULT_PATH).exists():
        with open(BEST_RESULT_PATH) as f:
            best_params = json.load(f)["config"]
        print("best config (from saved json):", best_params)
    else:
        print("Warning: No best result found. Try to run with --neps to find the best config")
        exit(0)
        
    # load the dataset and create a loader then pass it
    automl = AutoML(seed=seed)

    automl.fit(dataset_class, epochs=1, cfg=best_params)
    
    test_preds, test_labels = automl.predict(dataset_class)

    # Write the predictions of X_test to disk
    # This will be used by github classrooms to get a performance
    # on the test set.
    logger.info("Writing predictions to disk")
    with output_path.open("wb") as f:
        np.save(f, test_preds)

    # check if test_labels has missing data
    if not np.isnan(test_labels).any():
        acc = accuracy_score(test_labels, test_preds)
        logger.info(f"Accuracy on test set: {acc}")
    else:
        # This is the setting for the exam dataset, you will not have access to the labels
        logger.info(f"No test split for dataset '{dataset}'")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="The name of the dataset to run on.",
        choices=["fashion", "flowers", "emotions"]
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
        help="Use best NEPS config for training instead of Optuna"
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
        use_neps=args.neps
    )