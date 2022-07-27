from typing import Optional, Tuple, Union
from pathlib import Path

from nnood.utils.file_operations import save_json


def generate_dataset_json(output_file: Union[str, Path], train_dir: Union[str, Path],
                          test_dir: Optional[Union[str, Path]], modalities: Tuple[str], dataset_name: str,
                          licence: str = 'hands off!', dataset_description: str = '', dataset_reference='',
                          dataset_release: str = '0.0', data_augs: dict = {}, has_uniform_background: bool = False):
    """
    :param output_file: This needs to be the full path to the dataset.json you intend to write, so
    output_file='DATASET_PATH/dataset.json' where the folder DATASET_PATH points to is the one with the
    imagesTr and optional imagesTs.
    :param train_dir: Path to the imagesTr folder of that dataset.
    :param test_dir: Path to the imagesTs folder of that dataset. Can be None.
    :param modalities: Tuple of strings with modality names. must be in the same order as the images (first entry
    corresponds to _0000.nii.gz, etc). Example: ('T1', 'T2', 'FLAIR').
    :param dataset_name: The name of the dataset.
    :param licence: Dataset licence.
    :param dataset_description: Brief description of dataset.
    :param dataset_reference: Website of the dataset, if available.
    :param dataset_release:
    :param data_augs: Dictionary of valid data augmentations for data (which keep samples within normal distribution).
    :param has_uniform_background: Whether images have a uniform background. If true, the preprocessing creates a
    foreground mask. In this process it assumes the corners of the image are included in the background.
    :return:
    """

    train_path = Path(train_dir)
    test_path = Path(test_dir)

    train_ids = set(['_'.join(f.name.split('_')[:-1]) for f in train_path.iterdir() if f.is_file()])
    test_ids = set(['_'.join(f.name.split('_')[:-1]) for f in test_path.iterdir() if f.is_file()])

    dataset_json = {
        'name': dataset_name,
        'description': dataset_description,
        'reference': dataset_reference,
        'licence': licence,
        'release': dataset_release,
        'tensorImageSize': '3D' if any('png' in m for m in modalities) else '4D',
        'modality': {str(i): modalities[i] for i in range(len(modalities))},
        'numTraining': len(train_ids),
        'numTest': len(test_ids),
        'training': [ident for ident in train_ids],
        'test': [ident for ident in test_ids],
        'data_augs': data_augs,
        'has_uniform_background': has_uniform_background
    }
    save_json(dataset_json, output_file)
