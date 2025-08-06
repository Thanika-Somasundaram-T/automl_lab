# AutoML Exam - SS25 (Vision Data)
## Installation

To install the repository, first create an environment of your choice and activate it. 

For example, using `venv`:

You can change the python version here to the version you prefer.

**Virtual Environment**

```bash
python3 -m venv automl-vision-env
source automl-vision-env/bin/activate
```

**Conda Environment**

Can also use `conda`, left to individual preference.

```bash
conda create -n automl-vision-env python=3.11
conda activate automl-vision-env
```

Then install the repository by running the following command:

```bash
pip install -e .
```

You can test that the installation was successful by running the following command:

```bash
python -c "import automl"
```

We make no restrictions on the python library or version you use, but we recommend using python 3.8 or higher.

## Code

We provide the following:

* `main.py`: A script that trains an _AutoML-System_ on the training split `dataset_train` of a given dataset and then
  generates predictions for the test split `dataset_test`, saving those predictions to a file. For the training
  datasets`Flowers, Emotions, Fashion`, the test splits will contain the ground truth labels, but for the test dataset `Skin Cancer` the
  labels of the test split will not be available. You will be expected to generate these labels yourself and submit
  them to us through GitHub classrooms.

## Data

The datasets can be automatically downloaded by the
respective dataset classes in `./src/automl/datasets.py`. The datasets are: _fashion_, _flowers_, and _emotions_.

If there are any problems downloading the datasets, you can download them manually:

- Practice datasets : https://ml.informatik.uni-freiburg.de/research-artifacts/automl-exam-25-vision/vision-phase1.zip
- Final test dataset : https://ml.informatik.uni-freiburg.de/research-artifacts/automl-exam-25-vision/vision-phase2.zip

After downloading, unzip them and place the contents in the `/data` folder.

The downloaded datasets will have the following structure:
```bash
./data
├── fashion
│   ├── images_test
│   │   ├── 000001.jpg
│   │   ├── 000002.jpg
│   │   ├── 000003.jpg
│   │   ...
│   ├── images_train
│   │   ├── 000001.jpg
│   │   ├── 000002.jpg
│   │   ├── 000003.jpg
│   │   ...
│   ├── description.md
│   ├── test.csv
│   └── train.csv
├── emotions
    ...
...
```

| Dataset name | # Classes | # Train samples | # Test samples | # Channels | Resolution | Reference Accuracy | Our final Accuracy
|--------------|-----------|-----------------|----------------|------------|------------|--------------------|--------------------|
| fashion      | 10        | 60,000          | 10,000         | 1          | 28x28      | 0.88               | 0.90
| flowers      | 102*      | 5732            | 2,457          | 3          | 512x512    | 0.55               | 0.46
| emotions     | 7         | 28709           | 7,178          | 1          | 48x48      | 0.40               | 0.61
| **skin_cancer**  |**7\***       | **7,010**          | **3,005**          | **3**          | **450x450**    | **0.71**               | **0.76**

*classes are imbalanced

The final test dataset is `skin_cancer`along with its class definition to the `datasets.py` file. 
The test dataset is in the same
format as the training datasets, but `test.csv` will only contain nan's for labels.



## Running an initial test

This will download the _fashion_ dataset into `./data`, train a dummy AutoML system and generate predictions for the test
split:

```bash 
python main.py --dataset fashion --seed 42 --output-path preds-42-fashion.npy
```

You are free to modify these files and command line arguments as you see fit.

```
```

<span style="color:red"> **Note that any edits to the yaml workflow script are prohibited and monitored!** </span>

## Final submission

The following must be submitted by `August 6, 2025, 23:59 CET` for a successful project submission and poster participation:

#### **1) Poster submission**
Upload your poster as a PDF file named as `final_poster_vision_<team-name>.pdf`, following the template given [here](https://docs.google.com/presentation/d/1T55GFGsoon9a4T_oUm4WXOhW8wMEQL3M/edit?slide=id.p1#slide=id.p1).

#### **2) Test predictions**
The final test predictions should be uploaded in a file `final_test_preds.npy`, with each line containing the predictions for the input in the exact order of `X_test` given.

#### **3) Reproducibility instructions**
TL;DR: Code and instructions to _reproduce_ the above test predictions.

A `run_instructions.md` file that guides through the command to run the designed AutoML solution on the training set of the *final-test-dataset*.
This command should return either a: (i) hyperparameter configuration, (ii) a partially trained model on a hyperparameter configuration, or (iii) a fully trained model in `24 hours` at most.
A second command that given (i), (ii), or (iii) would do the needful that yields predictions for `test_X` for the *final-test-dataset*. This is the `final_test_preds.npy`.

#### **4) Team information**
Upload a file `team_info.txt` with the list of matriculation IDs of team members (*NO NAMES*). (E.g.: 1234567, 7654321)

### Submission checklist:
- [ ] Poster
- [ ] Test predictions
- [ ] Reproducibility instructions
- [ ] Team info
- [x] *Example to denote task being done*
<!-- This is a comment. -->


## Tips

* If you need to add dependencies that you and your teammates are all on the same page, you can modify the
  `pyproject.toml` file and add the dependencies there. This will ensure that everyone has the same dependencies

* Please feel free to modify the `.gitignore` file to exclude files generated by your experiments, such as models,
  predictions, etc. Also, be friendly teammate and ignore your virtual environment and any additional folders/files
  created by your IDE.
