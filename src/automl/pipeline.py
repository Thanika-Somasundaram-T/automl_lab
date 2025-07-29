import json
import time
from pathlib import Path
from automl.automl import AutoML
from automl.train import train_and_validate
from automl.utils import set_global_seed

def neps_training_wrapper(dataset_class, seed, neps_dir=Path):
    BEST_RESULT_PATH = neps_dir / "neps_best_result.json"
    def evaluate_pipeline(**config):
        set_global_seed(seed)

        start_time = time.time()

        # Train and validate
        val_acc = train_and_validate(config, dataset_class=dataset_class, seed=seed)
        elapsed_time = time.time() - start_time

        # Load best accuracy so far (from file)
        best_acc = -1.0
        if Path(BEST_RESULT_PATH).exists():
            try:
                with open(BEST_RESULT_PATH, "r") as f:
                    best_acc = json.load(f).get("val_acc", -1.0)
            except Exception as e:
                print(f"[NEPS] Warning: Could not load best result file: {e}")

        # Save if this is the new best
        if val_acc > best_acc:
            Path(BEST_RESULT_PATH).parent.mkdir(parents=True, exist_ok=True)
            with open(BEST_RESULT_PATH, "w") as f:
                json.dump({"config": config, "val_acc": val_acc}, f)
            print(f"[NEPS] New best model! val_acc={val_acc:.4f} (improved from {best_acc:.4f})")

        return {
            "objective_to_minimize": 1 - val_acc,
            "cost": elapsed_time,
            "info_dict": {
                "val_acc": val_acc,
                "training_time": elapsed_time,
                "training_time_min": elapsed_time / 60,
                "training_time_hr": elapsed_time / 3600,
                "model": config.get("model"),
                "head_layers": config.get("head_layers"),
                "batch_size": config.get("batch_size"),
                "optimizer": config.get("optimizer"),
                "lr": config.get("lr"),
                "hidden_dim": config.get("hidden_dim"),
                "dropout": config.get("dropout"),
                "max_epochs": config.get("max_epochs"),
            }
        }

    return evaluate_pipeline
