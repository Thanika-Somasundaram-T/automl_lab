import json
import time
from pathlib import Path

import torch
from automl.utils import set_global_seed
from automl.fit1 import fit1
def neps_phase1_wrapper(loaders, seed, neps_dir=Path):
    
    def evaluate_pipeline(**config):
        
        set_global_seed(seed)
        trial_name = config.get("_trial_id", f"trial_{int(time.time())}")

        start_time = time.time()
           
        result = fit1(config, loaders=loaders, trial_name=trial_name, seed=seed)
        
        elapsed_time = time.time() - start_time
        
        val_acc = result["val_acc"]

        # Path for saving best results
        best_result_path = neps_dir / "neps_best_result.json"
        best_model_path = neps_dir / "best_model.pth"
        best_config_path = neps_dir / "best_config.json"
        
        if best_result_path.exists():
            with open(best_result_path, "r") as f:
                best_data = json.load(f)
        else:
            best_data = {"val_acc": 0}

        # Update best if improved
        if val_acc > best_data.get("val_acc", 0):
            print("NEW BEST CONFIG FOUND WITH NEW ACC: ", val_acc, "OLD ACC: ", best_data.get("val_acc", 0))
            best_data = {
                "val_acc": val_acc,
                "training_time": elapsed_time,
                "config": config
            }
            print()
            print(best_data)
            print("*******************************************")
            # Save best model state dict
            torch.save(result["best_model_state"], best_model_path)
            # Save best config json
            with open(best_config_path, "w") as f:
                json.dump(config, f, indent=2)
            # Save best result json
            with open(best_result_path, "w") as f:
                json.dump(best_data, f, indent=2)

        return {
            "objective_to_minimize": 1.0 - result["val_acc"],
            "cost": elapsed_time,
            "info_dict": {
                "val_acc": result["val_acc"],
                "training_time": elapsed_time,
                "training_time_min": elapsed_time / 60,
                "training_time_hr": elapsed_time / 3600,
                "max_epochs": config.get("max_epochs"),
            }
        }

    return evaluate_pipeline
