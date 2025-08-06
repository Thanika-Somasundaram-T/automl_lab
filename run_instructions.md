## Kindly read the README.md file before runnig

The entire pipeline comprises `three` phases where
`Phase 1` trains with 3 train datasets combined.

`Phase 2` and `Phase 3` train and predict depends on the dataset provided in the command/terminal
as `--dataset dataset_name`

### **To run the Entire Pipeline for the test dataset**

```bash
python3 main.py --dataset skin_cancer
```
### **To run the pipeline with custom dataset**

```bash
python3 main.py --dataset dataset_name
```

### **Possible args you can give**
`--dataset dataset_name`

`--search_dir phase1_directory`

`--finetune_dir phase2_directory`

`--output_path final_prediction.py_directory`

`--seed 42`

```bash
python3 main.py --dataset dataset_name --search_dir phase1_directory --finetune_dir phase2_directory --output_path final_prediction.py_directory --seed 42
```

