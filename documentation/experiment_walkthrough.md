
# Experiment walkthrough

Before running any programs, define nnOOD's enviroment variables:
 - `nnood_raw_data_base` - Folder containing raw data of each dataset.
 - `nnood_preprocessed_data_base` - Folder containing preprocessed datasets and experiment plans.
 - `nnood_results_base` - Folder containing outputs of experiments: trained models, logs and any test results.

## Dataset conversion

nnOOD uses a similar dataset structure to nnU-Net. Each dataset has it's own folder within the raw dataset folder,
`$nnood_raw_data_base/DATASET_NAME`, which must contain:
  - `imagesTr/` folder, containing normal training images. Images must follow naming convention
  `<sample_id>_MMMM.[png|nii.gz]`, where MMMM is the modality number.
  - `imagesTs` folder, containing test images, following the same naming convention as `imagesTr`.
  - `labelsTs` folder, containing the labels for the test images. Test images must follow naming convention
  `<sample_id>.[png|nii.gz]`. If a label is not provided for a test image, it is assumed to be entirely normal.
  - `dataset.json` file, describing dataset, described in `nnood/data/readme.md`
    - We recommend using `nnood/data/dataset_conversion/utils.generate_dataset_json` to produce this.

## Experiment planning and preprocessing

As with nnU-Net, nnOOD xtracts a dataset fingerprint, creates an experiment plan and preprocesses the dataset ready for
training.

```bash
python <path_to_project>/nnood/experiment_planning/nnOOD_plan_and_preprocess.py -d DATASET_NAME --verify_dataset_integrity
```

This stores the experiment plans and data within `$nnood_preprocessed_data_base/DATASET_NAME`.
The experiment plan is independent to the self-supervised tasks.

## Model training

nnOOD uses 5-fold cross-validation to try to get a more consistent measurement of a task on a given dataset.
To train one fold of this proces, use the command:

```bash
python <path_to_project>/nnood/training/nnOOD_run_training.py CONFIGURATION TRAINER_CLASS_NAME DATASET_NAME TASK_NAME FOLD
```

CONFIGURATION can currently only be `lowres` or `fullres`, depending on whether the dataset recommends one or two
stages, as at this point we have not been able to implement cascade training.

TRAINER_CLASS_NAME chooses between `nnOODTrainer` and `nnOODTrainerDS`, which determines whether deep supervision is
used, although based on nnU-Net's results we recommend using `nnOODTrainerDS` unless the user needs to be especially
careful with memory consumption.

TASK_NAME is the name of the class which extends and implements `SelfSupTask` (such as `FPI`, `CutPaste`, etc) and
must be stored within `nnood/self_supervised_task`.

Other options can be viewed using the `-h` flag.

The outputs of this experiment are stored in:
```
$nnood_results_base/DATASET_NAME/TASK_NAME/CONFIGURATION/<TRAINER_CLASS_NAME>_<default_plans_identifier>/fold_FOLD/
```
where `default_plans_identifier` is defined in `nnood/paths.py`.

The outputs are:
 - `training_log_<experiment_datetime>.txt` - log of outputs during experiment
 - `model_(best|final_checkpoint).(model|pkl)` - files containing trained model and training progress

## Testing models

Test the ensemble of the 5 models trained as part of the cross-validation on a certain task with the command:
```bash
python <path_to_project>/nnood/evaluation/nnOOD_run_testing.py -d DATASET_NAME -t TASK_NAME -tr TRAINER_CLASS_NAME
```

Other options can be viewed using the `-h` flag.

The models are evaluated on the datasets test set, given in `imagesTs` and `labelsTs`, as described in
[dataset conversion](#dataset-conversion).

This computes the following:
 - A prediction for each test image, saved as `<sample_id>.[png|nii.gz]`
 - `summary.json` - a record of the metrics on the test dataset (AUROC, AP) along with a timestamp for when the test
  took place.

All are saved to:
```
$nnood_results_base/DATASET_NAME/TASK_NAME/testResults/<default_plans_identifier>/
```