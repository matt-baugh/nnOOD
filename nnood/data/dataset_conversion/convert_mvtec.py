from typing import Union
import sys
import os
from pathlib import Path
import shutil

from nnood.paths import raw_data_base, DATASET_JSON_FILE
from nnood.data.dataset_conversion.utils import generate_dataset_json

CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
               'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
               'tile', 'toothbrush', 'transistor', 'wood', 'zipper']

OBJECTS = ['bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut',
           'pill', 'screw', 'toothbrush', 'transistor', 'zipper']
TEXTURES = ['carpet', 'grid', 'leather', 'tile', 'wood']

HAS_UNIFORM_BACKGROUND = ['bottle', 'capsule', 'hazelnut', 'metal_nut', 'pill', 'screw', 'toothbrush', 'zipper']

# Bottle is aligned but it's symmetric under rotation
UNALIGNED_OBJECTS = ['bottle', 'hazelnut', 'metal_nut', 'screw']

H_FLIP = TEXTURES + ['bottle', 'hazelnut', 'toothbrush', 'transistor', 'zipper']
V_FLIP = TEXTURES + ['bottle', 'hazelnut']
ROTATE = UNALIGNED_OBJECTS + ['leather', 'tile', 'grid']
SMALL_ROTATE = ['carpet', 'wood']

GREYSCALE = ['grid', 'screw', 'zipper']


for g in [OBJECTS, TEXTURES, UNALIGNED_OBJECTS, H_FLIP, V_FLIP, ROTATE, SMALL_ROTATE]:
    assert all([e in CLASS_NAMES for e in g]), f'Element of {g} not in CLASS_NAMES'


def organise_class(in_dir: Union[str, Path]):

    assert os.path.isdir(in_dir), 'Not a valid directory: ' + in_dir
    in_dir_path = Path(in_dir)

    in_train_path = in_dir_path / 'train' / 'good'
    assert in_train_path.is_dir()

    in_test_examples_path = in_dir_path / 'test'
    assert in_test_examples_path.is_dir()

    in_test_labels_path = in_dir_path / 'ground_truth'
    assert in_test_labels_path.is_dir()

    test_dirs = [d for d in in_test_examples_path.iterdir() if d.is_dir()]
    assert len(test_dirs) > 1, 'Test must include good and bad examples'

    object_class = in_dir_path.name

    out_dir_path = Path(raw_data_base) / ('mvtec_ad_' + object_class)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    out_train_path = out_dir_path / 'imagesTr'
    out_train_path.mkdir(parents=True, exist_ok=True)

    # Copy normal training data
    for f in in_train_path.iterdir():
        file_name = f.name
        number, ext = file_name.split('.')
        shutil.copy(f, out_train_path / f'normal_{number}_0000.{ext}')

    out_test_path = out_dir_path / 'imagesTs'
    out_test_path.mkdir(parents=True, exist_ok=True)

    out_test_labels_path = out_dir_path / 'labelsTs'
    out_test_labels_path.mkdir(parents=True, exist_ok=True)

    # Copy testing data
    for d in test_dirs:
        folder_name = d.name

        if folder_name != 'good':
            # Check labels folder exists
            test_class_label_dir = in_test_labels_path / folder_name
            assert test_class_label_dir.is_dir(), 'Missing labels folder: ' + test_class_label_dir.__str__()

        for f in d.iterdir():
            file_name = f.name
            id_num, ext = file_name.split('.')

            if folder_name != 'good':
                # Verify and copy label for test example
                test_label_path = in_test_labels_path / folder_name / f'{id_num}_mask.{ext}'
                shutil.copy(test_label_path, out_test_labels_path / f'{folder_name}_{id_num}.{ext}')

            shutil.copy(f, out_test_path / f'{folder_name}_{id_num}_0000.{ext}')

    data_augs = {}

    if object_class in ROTATE:
        data_augs['rotation'] = {'rot_max': 5}
    elif object_class in SMALL_ROTATE:
        data_augs['rotation'] = {'rot_max': 2}

    if object_class in H_FLIP or object_class in V_FLIP:
        axes = []
        if object_class in V_FLIP:
            axes.append(0)

        if object_class in H_FLIP:
            axes.append(1)

        data_augs['mirror'] = {'mirror_axes': axes}

    png_type = 'png-bw' if object_class in GREYSCALE else 'png'

    generate_dataset_json(out_dir_path / DATASET_JSON_FILE, out_train_path, out_test_path, (png_type,), in_dir_path.name,
                          licence='CC BY-NC-SA 4.0',
                          dataset_description='Images from the MVTec Anomaly Detection Dataset for the class ' +
                                              in_dir_path.name, dataset_reference='MVTec Software GmbH',
                          dataset_release='1.0 16/04/2021', data_augs=data_augs,
                          has_uniform_background=object_class in HAS_UNIFORM_BACKGROUND)


if __name__ == '__main__':
    # Folder of image class, or root mvtec dataset folder
    in_root_dir: str = sys.argv[1]

    if len(sys.argv) == 3 and sys.argv[2] == 'full_dataset':
        print('Processing entire MVTec AD Dataset')
        in_root_path = Path(in_root_dir)

        for in_class_dir in in_root_path.iterdir():
            if not in_class_dir.is_dir():
                continue

            print(f'Processing {in_class_dir}...')

            organise_class(in_class_dir)

    else:
        print('Processing single class...')
        organise_class(in_root_dir)
    print('Done!')
