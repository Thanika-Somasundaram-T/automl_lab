
import json
from pathlib import Path
from automl.automl import AutoML

BEST_RESULT_PATH = "neps_results/best_result.json"
LOSSES_LOG_PATH = Path("neps_results/losses_log.json")


def neps_training_wrapper(dataset_class, seed):
    def evaluate_pipeline(**config):
        automl = AutoML(seed=seed)
        automl.fit(dataset_class, epochs=15, cfg=config, is_val=True)
        train_loss = automl.train_losses
        val_loss = automl.val_losses
        
        val_error = 1 - automl.accuracy

        print("Ended: ", val_error, "Val_acc: ", automl.accuracy)

        # Save best result
        if not Path(BEST_RESULT_PATH).exists():
            current_best = {"val_error": 1.0}
        else:
            with open(BEST_RESULT_PATH) as f:
                current_best = json.load(f)

        if val_error < current_best["val_error"]:
            with open(BEST_RESULT_PATH, "w") as f:
                json.dump({"val_error": val_error, "val_acc": automl.accuracy, "config": config}, f, indent=2)
                
        LOSSES_LOG_PATH.parent.mkdir(exist_ok=True)
        if LOSSES_LOG_PATH.exists():
            with open(LOSSES_LOG_PATH) as f:
                all_losses = json.load(f)
        else:
            all_losses = {"train": [], "val": []}

        all_losses["train"].append(train_loss)
        all_losses["val"].append(val_loss)

        with open(LOSSES_LOG_PATH, "w") as f:
            json.dump(all_losses, f, indent=2)

        return {
            "objective_to_minimize": val_error,
            "cost": 0,
            "info_dict": {"val_acc": automl.accuracy}
        }
    return evaluate_pipeline
