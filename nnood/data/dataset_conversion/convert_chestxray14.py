
import os
from pathlib import Path
from typing import Union, Optional

import cv2
import numpy as np
import pandas as pd

from nnood.data.dataset_conversion.utils import generate_dataset_json
from nnood.paths import raw_data_base, DATASET_JSON_FILE

# Dataset available at:
# https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938765345

script_dir = Path(os.path.realpath(__file__)).parent

xray_list_dir = script_dir / 'chestxray14_lists'

train_male_list_path = xray_list_dir / 'norm_MaleAdultPA_train_list.txt'
test_male_list_path = xray_list_dir / 'anomaly_MaleAdultPA_test_list.txt'

train_female_list_path = xray_list_dir / 'norm_FemaleAdultPA_train_list.txt'
test_female_list_path = xray_list_dir / 'anomaly_FemaleAdultPA_test_list.txt'

bbox_data_file_path = xray_list_dir / 'BBox_List_2017.csv'
bbox_csv = pd.read_csv(bbox_data_file_path, index_col=0, usecols=['Image Index', 'Bbox [x', 'y', 'w', 'h]'])

train_test_dict = {
    'male': (train_male_list_path, test_male_list_path),
    'female': (train_female_list_path, test_female_list_path)
}


def organise_xray_data(in_dir: Union[str, Path], data_type: str):

    in_dir_path = Path(in_dir)
    assert in_dir_path.is_dir(), 'Not a valid directory: ' + in_dir

    train_list_path, test_list_path = train_test_dict[data_type]

    out_dir_path = Path(raw_data_base) / f'chestXray14_PA_{data_type}'
    out_dir_path.mkdir(parents=True, exist_ok=True)

    out_train_path = out_dir_path / 'imagesTr'
    out_train_path.mkdir(parents=True, exist_ok=True)

    def load_and_save(f_name: str, curr_out_dir_path: Path, mask_out_dir_path: Optional[Path]):
        f_path = in_dir_path / f_name
        assert f_path.is_file(), f'Missing file: {f_path}'

        opencv_img = cv2.imread(f_path.__str__(), cv2.IMREAD_GRAYSCALE)
        assert len(opencv_img.shape) == 2, 'Greyscale shape not 2?? ' + opencv_img.shape

        sample_id = f_path.stem

        f_out_path = curr_out_dir_path / (sample_id + '_0000.png')
        cv2.imwrite(f_out_path.__str__(), opencv_img)

        if mask_out_dir_path is not None:
            assert f_name in bbox_csv.index, 'Missing bbox data for ' + f_name
            sample_mask = np.zeros_like(opencv_img)

            curr_mask_data = bbox_csv.loc[f_name].to_numpy().round().astype(int)
            if len(curr_mask_data.shape) == 1:
                curr_mask_data = [curr_mask_data]

            for bbox_x, bbox_y, bbox_w, bbox_h in curr_mask_data:
                sample_mask[bbox_y: bbox_y + bbox_h, bbox_x: bbox_x + bbox_w] = 1

            mask_path = mask_out_dir_path / f_name
            cv2.imwrite(mask_path.__str__(), sample_mask)

        return sample_id

    # # Load healthy files
    # with open(train_list_path, 'r') as train_list_file:
    #     for f in train_list_file.readlines():
    #         load_and_save(f.strip(), out_train_path, None)

    out_test_path = out_dir_path / 'imagesTs'
    out_test_path.mkdir(parents=True, exist_ok=True)

    out_test_labels_path = out_dir_path / 'labelsTs'
    out_test_labels_path.mkdir(parents=True, exist_ok=True)

    # # Load test files
    # with open(test_list_path, 'r') as test_list_file:
    #     for f in test_list_file.readlines():
    #         f = f.strip()
    #         # Only include in test set if has bounding box
    #         if f in bbox_csv.index:
    #             load_and_save(f.strip(), out_test_path, out_test_labels_path)

    data_augs = {
        'scaling': {'scale_range': [0.97, 1.03]}
    }

    generate_dataset_json(out_dir_path / DATASET_JSON_FILE, out_train_path, out_test_path, ('png-xray',),
                          out_dir_path.name,
                          dataset_description='Images from the NIH Chest X-ray dataset; limited to posteroanterior '
                                              f'views of {data_type} adult patients (over 18), with the test set only '
                                              'including patients which had a bounding box provided.',
                          data_augs=data_augs)

# CHANGE THESE TO MATCH YOUR DATA!!!
organise_xray_data('/vol/biodata/data/chest_xray/ChestXray-NIHCC/images', 'male')
organise_xray_data('/vol/biodata/data/chest_xray/ChestXray-NIHCC/images', 'female')
