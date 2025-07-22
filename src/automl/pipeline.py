
import json
from pathlib import Path
from automl.automl import AutoML

BEST_RESULT_PATH = "neps_results/best_result.json"


def neps_training_wrapper(dataset_class, seed):
    def evaluate_pipeline(**config):
        automl = AutoML(seed=seed)
        acc = automl.fit(dataset_class, epochs=1, cfg=config, is_val=True)
        val_error = 1 - acc

        print("Ended: ", val_error, "Val_acc: ", acc)

        # Save best result
        if not Path(BEST_RESULT_PATH).exists():
            current_best = {"val_error": 1.0}
        else:
            with open(BEST_RESULT_PATH) as f:
                current_best = json.load(f)

        if val_error < current_best["val_error"]:
            with open(BEST_RESULT_PATH, "w") as f:
                json.dump({"val_error": val_error, "val_acc": acc, "config": config}, f, indent=2)

        return {
            "objective_to_minimize": val_error,
            "cost": 0,
            "info_dict": {"val_acc": acc}
        }
    return evaluate_pipeline
