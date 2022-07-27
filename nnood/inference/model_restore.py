from pathlib import Path

import torch

import nnood
from nnood.training.network_training.nnOODTrainer import nnOODTrainer
from nnood.utils.file_operations import load_pickle
from nnood.utils.miscellaneous import recursive_find_python_class


def restore_model(pkl_file: Path, checkpoint=None, train=False, fp16=None) -> nnOODTrainer:
    """
    This is a utility function to load any nnOODTrainer from a pkl. It will recursively search
    nnood.training.network_training for the file that contains the trainer and instantiate it with the arguments saved
    in the pkl file. If checkpoint is specified, it will furthermore load the checkpoint file in train/test mode (as
    specified by train). The pkl file required here is the one that will be saved automatically when calling
    nnOODTrainer.save_checkpoint.
    :param pkl_file:
    :param checkpoint:
    :param train:
    :param fp16: if None then we take no action. If True/False we overwrite what the model has in its init
    :return:
    """
    info = load_pickle(pkl_file)
    init = info['init']
    trainer_class_name = info['name']

    trainer_class = recursive_find_python_class([Path(nnood.__path__[0], 'training', 'network_training').__str__()],
                                                trainer_class_name, current_module='nnood.training.network_training')

    if trainer_class is None:
        raise RuntimeError('Could not find the model trainer specified in checkpoint in'
                           'nnood.training.network_training. If it is not locatd there, please move it or change the'
                           'code of restore_model. Your model can be located in any directory within'
                           'nnood.training.network_training (search is recursive. \n Debug info: \n checkpoint file:'
                           f'{checkpoint}\nName of trainer: {trainer_class_name}')

    assert issubclass(trainer_class, nnOODTrainer), 'The network trainer was found but is not a subclass of ' \
                                                    'nnOODTrainer. Please make it so!'

    # From nnUNet, meaning lost to time:
    # ToDo Fabian make saves use kwargs, please...

    trainer = trainer_class(*init)

    # We can hack fp16 overwriting into the trainer without changing the init arguments because nothing happens with
    # fp16 in the init, it just saves it to a member variable
    if fp16 is not None:
        trainer.fp16 = fp16

    trainer.process_plans(info['plans'])
    if checkpoint is not None:
        trainer.load_checkpoint(checkpoint, train)
    return trainer


def load_model_and_checkpoint_files(folder: Path, folds=None, mixed_precision=None, checkpoint_name='model_best'):
    """
    used for if you need to ensemble the five models of a cross-validation. This will restore the model from the
    checkpoint in fold 0, load all parameters of the five folds in ram and return both. This will allow for fast
    switching between parameters (as opposed to loading them form disk each time).
    This is best used for inference and test prediction
    :param folder:
    :param folds:
    :param mixed_precision: if None then we take no action. If True/False we overwrite what the model has in its init
    :param checkpoint_name:
    :return:
    """
    if isinstance(folds, str):
        folds = [folder / 'all']
        assert folds[0].is_dir(), f'no output folder for fold {folds} found'
    elif isinstance(folds, (list, tuple)):
        if len(folds) == 1 and folds[0] == 'all':
            folds = [folder / 'all']
        else:
            folds = [folder / f'fold_{i}' for i in folds]
    elif isinstance(folds, int):
        folds = [folder / f'fold_{folds}']
    elif folds is None:
        print('folds is None so we will automatically look for output folders (not using \'all\'!)')
        folds = [f for f in folder.iterdir() if f.is_dir() and f.name.startswith('fold')]
        print('found the following folds: ', folds)
    else:
        raise ValueError(f'Unknown value for folds. Type: {type(folds)}. Expected: list of int, int, str or None')

    assert all([f.is_dir() for f in folds]), 'list of folds specified but not all output folders are present'

    trainer = restore_model(folds[0] / f'{checkpoint_name}.pkl', fp16=mixed_precision)
    trainer.output_folder = folder
    trainer.output_folder_base = folder
    # I think fold is set as otherwise load_best_checkpoint raises an exception, even though trainer.fold isn't used
    # during inference
    trainer.update_fold(0)
    trainer.initialize(False)
    all_model_files = [f / f'{checkpoint_name}.model' for f in folds]
    print('using the following model files: ', all_model_files)
    all_params = [torch.load(i, map_location=torch.device('cpu')) for i in all_model_files]
    return trainer, all_params
