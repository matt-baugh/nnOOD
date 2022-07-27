import numpy as np
from pathlib import Path
from typing import List

import nibabel as nib
import SimpleITK as sitk

from nnood.utils.file_operations import load_json


def verify_all_same_orientation(all_files: List[Path]):

    # Assumes files are either .nii.gz or .png
    nii_files = [f for f in all_files if f.suffix != '.png']
    orientations = []
    for n in nii_files:
        img = nib.load(n)
        affine = img.affine
        orientation = nib.aff2axcodes(affine)
        orientations.append(orientation)
    # now we need to check whether they are all the same
    orientations = np.array(orientations)
    unique_orientations = np.unique(orientations, axis=0)
    all_same = len(unique_orientations) == 1
    return all_same, unique_orientations


def verify_same_geometry(img_1: sitk.Image, img_2: sitk.Image):
    ori1, spacing1, direction1, size1 = img_1.GetOrigin(), img_1.GetSpacing(), img_1.GetDirection(), img_1.GetSize()
    ori2, spacing2, direction2, size2 = img_2.GetOrigin(), img_2.GetSpacing(), img_2.GetDirection(), img_2.GetSize()

    same_ori = np.all(np.isclose(ori1, ori2))
    if not same_ori:
        print('The origin does not match between the images:')
        print(ori1)
        print(ori2)

    same_spac = np.all(np.isclose(spacing1, spacing2))
    if not same_spac:
        print('The spacing does not match between the images')
        print(spacing1)
        print(spacing2)

    same_dir = np.all(np.isclose(direction1, direction2))
    if not same_dir:
        print('The direction does not match between the images')
        print(direction1)
        print(direction2)

    same_size = np.all(np.isclose(size1, size2))
    if not same_size:
        print('The size does not match between the images')
        print(size1)
        print(size2)

    if same_ori and same_spac and same_dir and same_size:
        return True
    else:
        return False


def verify_dataset_integrity(dataset_folder: Path):
    """
    folder needs the imagesTr folder and dataset.json.
    Optional imagesTs and labelsTr folders.
    Checks if all training cases and labels are present.
    Checks if all test cases (if any) are present.
    For each case, checks whether all modalities are present.
    For each case, checks whether the pixel grids are aligned.
    :param dataset_folder:
    :return:
    """
    print('Verifying dataset in ', dataset_folder)

    assert dataset_folder.is_dir(), 'Dataset folder doesn\'t exist!'

    dataset_file = dataset_folder / 'dataset.json'
    assert dataset_file.is_file(), 'Missing dataset.json'

    train_folder = dataset_folder / 'imagesTr'
    assert train_folder.is_dir(), 'Missing training folder imagesTr'

    dataset = load_json(dataset_file)
    modalities = dataset['modality']
    num_modalities = len(modalities)
    file_suffixes = ['png' if 'png' in modalities[str(i)] else 'nii.gz' for i in range(num_modalities)]

    training_ids = dataset['training']
    assert len(training_ids) == len(np.unique(training_ids)), 'Duplicate training ids in dataset.json!'
    training_files = [f for f in train_folder.iterdir() if f.is_file()]

    num_train_files = len(training_files)
    num_expected_train_files = num_modalities * len(training_ids)
    assert num_train_files <= num_expected_train_files, 'Extra files in training folder (should be just training ' \
                                                        f'data modalities): {num_train_files} > ' \
                                                        f'{num_expected_train_files}'
    assert num_train_files >= num_expected_train_files, f'Missing files in training folder: {num_train_files} < ' \
                                                        f'{num_expected_train_files}'

    test_ids = dataset['test']
    assert len(test_ids) == len(np.unique(test_ids)), 'Duplicate test ids in dataset.json!'

    geometries_OK = True
    has_nan = False
    all_files = []

    print('Verifying training set')
    for c in training_ids:
        print('Checking case', c)

        # Check if all files are present
        expected_image_files = [train_folder / f'{c}_{i:04d}.{file_suffixes[i]}' for i in range(num_modalities)]
        all_files += expected_image_files

        for f in expected_image_files:
            assert f in training_files, f'Missing file {f}'

        images_itk = [sitk.ReadImage(i.__str__()) for i in expected_image_files]

        for i, img in enumerate(images_itk):
            nans_in_image = np.any(np.isnan(sitk.GetArrayFromImage(img)))
            has_nan = has_nan | nans_in_image
            same_geometry = verify_same_geometry(img, images_itk[0])

            if not same_geometry:
                geometries_OK = False
                print(f'The geometry of the image {expected_image_files[i]} does not match the geometry of the label '
                      'file. The pixel arrays will not be aligned and nnU-Net cannot use this data. Please make sure '
                      'your image modalities are coregistered and have the same geometry as the label')

            if nans_in_image:
                print(f'There are NAN values in image {expected_image_files[i]}')

    # check test set, but only if there actually is a test set
    if len(test_ids) > 0:
        print('Verifying test set')

        test_folder = dataset_folder / 'imagesTs'
        assert test_folder.is_dir(), 'Test ids present, but no imagesTs folder!'

        test_labels_folder = dataset_folder / 'labelsTs'
        assert test_labels_folder.is_dir(), 'Test ids present, but no labelsTs folder! (if the examples are without ' \
                                            'ground truths, and you just want to predict them as examples, don\'t ' \
                                            'label them as tests'

        test_files = [f for f in test_folder.iterdir() if f.is_file()]
        num_test_files = len(test_files)
        num_expected_test_files = num_modalities * len(test_ids)
        assert num_test_files <= num_expected_test_files, 'Extra files in test folder (should be just test data ' \
                                                          f'modalities): {num_test_files} > {num_expected_test_files}'
        assert num_test_files >= num_expected_test_files, f'Missing files in training folder: {num_test_files} < ' \
                                                          f'{num_expected_test_files}'

        test_label_files = [f for f in test_labels_folder.iterdir() if f.is_file()]
        assert len(test_label_files) >= 1, 'Must have at least 1 ground truth label (cannot be all normal images)!'

        for c in test_ids:
            # Check if all files are present
            expected_image_files = [test_folder / f'{c}_{i:04d}.{file_suffixes[i]}' for i in range(num_modalities)]
            all_files += expected_image_files

            for f in expected_image_files:
                assert f in test_files, f'Missing file {f}'

            images_itk = [sitk.ReadImage(i.__str__()) for i in expected_image_files]

            files_to_check = images_itk

            # Bit of an assumption that label has same suffix as first image file
            # Unlikely that examples mix .png and .nii.gz
            curr_label_file = test_labels_folder / f'{c}.{file_suffixes[0]}'
            if curr_label_file in test_label_files:
                all_files.append(curr_label_file)
                files_to_check.append(sitk.ReadImage(curr_label_file.__str__()))

            for i, img in enumerate(files_to_check):
                nans_in_image = np.any(np.isnan(sitk.GetArrayFromImage(img)))
                has_nan = has_nan | nans_in_image
                same_geometry = verify_same_geometry(img, files_to_check[0])

                if not same_geometry:
                    geometries_OK = False
                    print(
                        f'The geometry of the image {expected_image_files[i]} does not match the geometry of the label '
                        'file. The pixel arrays will not be aligned and nnU-Net cannot use this data. Please make sure '
                        'your image modalities are coregistered and have the same geometry as the label')

                if nans_in_image:
                    print(f'There are NAN values in image {expected_image_files[i]}')

    all_same = verify_all_same_orientation(all_files)
    if not all_same:
        print('WARNING: Not all images in the dataset have the same axis ordering. We very strongly recommend you '
              'correct that by reorienting the data. fslreorient2std should do the trick')
    # save unique orientations to dataset.json
    if not geometries_OK:
        raise Warning('GEOMETRY MISMATCH FOUND! CHECK THE TEXT OUTPUT! This does not cause an error at this point but '
                      'you should definitely check whether your geometries are alright!')
    else:
        print('Dataset OK')

    if has_nan:
        raise RuntimeError('Some images have nan values in them. This will break the training. See text output above '
                           'to see which ones')
