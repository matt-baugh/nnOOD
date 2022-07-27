from collections import OrderedDict
from multiprocessing import Pool
from pathlib import Path
from typing import Tuple, Union

import SimpleITK as sitk
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from skimage.transform import resize

from nnood.configuration import default_num_processes, RESAMPLING_SEPARATE_Z_ANISO_THRESHOLD
from nnood.preprocessing.foreground_mask import get_object_mask
from nnood.preprocessing.normalisation import normalise
from nnood.utils.file_operations import save_pickle
from nnood.utils.miscellaneous import make_default_mask


def get_do_separate_z(spacing, anisotropy_threshold=RESAMPLING_SEPARATE_Z_ANISO_THRESHOLD):
    do_separate_z = (np.max(spacing) / np.min(spacing)) > anisotropy_threshold
    return do_separate_z


def get_lowres_axis(new_spacing):
    axis = np.where(max(new_spacing) / np.array(new_spacing) == 1)[0]  # find which axis is anisotropic
    return axis


def resample_patient(data, original_spacing, target_spacing, order=3, force_separate_z=False,
                     order_z=0, separate_z_anisotropy_threshold=RESAMPLING_SEPARATE_Z_ANISO_THRESHOLD):
    """
    :param data:
    :param original_spacing:
    :param target_spacing:
    :param order:
    :param force_separate_z: if None then we dynamically decide how to resample along z, if True/False then always
    /never resample along z separately
    :param order_z: only applies if force_separate_z is True
    :param separate_z_anisotropy_threshold: if max_spacing > separate_z_anisotropy_threshold * min_spacing (per axis)
    then resample along lowres axis with order_z_data/order_z_seg instead of order_data/order_seg
    :return:
    """

    assert len(data.shape) == 4 or len(data.shape) == 3, f'data must be c x y [z]: {data.shape}'

    shape = np.array(data[0].shape)

    new_shape = np.round(((np.array(original_spacing) / np.array(target_spacing)).astype(float) * shape)).astype(int)

    if force_separate_z is not None:
        do_separate_z = force_separate_z
        if force_separate_z:
            axis = get_lowres_axis(original_spacing)
        else:
            axis = None
    else:
        if get_do_separate_z(original_spacing, separate_z_anisotropy_threshold):
            do_separate_z = True
            axis = get_lowres_axis(original_spacing)
        elif get_do_separate_z(target_spacing, separate_z_anisotropy_threshold):
            do_separate_z = True
            axis = get_lowres_axis(target_spacing)
        else:
            do_separate_z = False
            axis = None

    if axis is not None:
        if len(axis) == len(shape):
            # every axis has the spacing, this should never happen, why is this code here?
            do_separate_z = False
        elif len(axis) == len(shape) - 1:
            # this happens for spacings like (0.24, 1.25, 1.25) for example. In that case we do not want to resample
            # separately in the out of plane axis
            do_separate_z = False

    data_reshaped = resample_data(data, new_shape, axis, order, do_separate_z, order_z=order_z)

    return data_reshaped


def resample_data(data, new_shape, axis=None, order=3, do_separate_z=False, order_z=0):
    """
    separate_z=True will resample with order 0 along z
    :param data:
    :param new_shape:
    :param axis:
    :param order:
    :param do_separate_z:
    :param order_z: only applies if do_separate_z is True
    :return:
    """
    assert len(data.shape) == 4 or len(data.shape) == 3, 'data must be (c, x, y[, z])'

    resize_kwargs = {'mode': 'edge', 'anti_aliasing': True}

    dtype_data = data.dtype
    shape = np.array(data[0].shape)
    new_shape = np.array(new_shape)
    if np.any(shape != new_shape):
        data = data.astype(float)
        if do_separate_z:
            assert len(shape) == 3, f'Should only need to perform separate z sampling on 3D data: {shape}'
            print('Separate z, order in z is', order_z, 'order inplane is', order)
            assert len(axis) == 1, 'only one anisotropic axis supported'
            axis = axis[0]
            if axis == 0:
                new_shape_2d = new_shape[1:]
            elif axis == 1:
                new_shape_2d = new_shape[[0, 2]]
            else:
                new_shape_2d = new_shape[:-1]

            reshaped_final_data = []
            for c in range(data.shape[0]):
                reshaped_data = []
                for slice_id in range(shape[axis]):
                    if axis == 0:
                        reshaped_data.append(resize(data[c, slice_id], new_shape_2d, order, **resize_kwargs))
                    elif axis == 1:
                        reshaped_data.append(resize(data[c, :, slice_id], new_shape_2d, order, **resize_kwargs))
                    else:
                        reshaped_data.append(resize(data[c, :, :, slice_id], new_shape_2d, order, **resize_kwargs))
                reshaped_data = np.stack(reshaped_data, axis)
                if shape[axis] != new_shape[axis]:

                    # The following few lines are blatantly copied and modified from sklearn's resize()
                    rows, cols, dim = new_shape[0], new_shape[1], new_shape[2]
                    orig_rows, orig_cols, orig_dim = reshaped_data.shape

                    row_scale = float(orig_rows) / rows
                    col_scale = float(orig_cols) / cols
                    dim_scale = float(orig_dim) / dim

                    map_rows, map_cols, map_dims = np.mgrid[:rows, :cols, :dim]
                    map_rows = row_scale * (map_rows + 0.5) - 0.5
                    map_cols = col_scale * (map_cols + 0.5) - 0.5
                    map_dims = dim_scale * (map_dims + 0.5) - 0.5

                    coord_map = np.array([map_rows, map_cols, map_dims])
                    reshaped_final_data.append(map_coordinates(reshaped_data, coord_map, order=order_z,
                                                               mode='nearest')[None])
                else:
                    reshaped_final_data.append(reshaped_data[None])
            reshaped_final_data = np.vstack(reshaped_final_data)
        else:
            print('No separate z, order', order)
            reshaped = []
            for c in range(data.shape[0]):
                reshaped.append(resize(data[c], new_shape, order, **resize_kwargs)[None])
            reshaped_final_data = np.vstack(reshaped)
        return reshaped_final_data.astype(dtype_data)
    else:
        print('No resampling necessary')
        return data


class GenericPreprocessor:
    def __init__(self, modalities, normalisation_scheme_per_modality, transpose_forward: (tuple, list),
                 intensity_properties, make_foreground_masks):

        self.modalities = modalities
        self.normalisation_scheme_per_modality = normalisation_scheme_per_modality
        self.transpose_forward = transpose_forward
        self.intensity_properties = intensity_properties
        self.make_foreground_masks = make_foreground_masks

        self.resample_separate_z_anisotropy_threshold = RESAMPLING_SEPARATE_Z_ANISO_THRESHOLD

    def load_and_combine(self, sample_identifier: str, input_folder: Path, return_properties: bool = False)\
            -> Union[Tuple[np.array, OrderedDict], np.array]:
        all_ndarrays = []
        properties = OrderedDict()
        properties['sample_id'] = sample_identifier
        properties['data_files'] = []

        for i in range(len(self.modalities)):
            mod = self.modalities[i]
            suffix = 'png' if 'png' in mod else 'nii.gz'
            file_path = input_folder / f'{sample_identifier}_{i:04d}.{suffix}'
            properties['data_files'].append(file_path)

            sitk_image = sitk.ReadImage(file_path.__str__())
            image_array = sitk.GetArrayFromImage(sitk_image).astype(np.float32)

            if i == 0:
                properties['original_size'] = np.array(sitk_image.GetSize())[::-1]
                properties['original_spacing'] = np.array(sitk_image.GetSpacing())[::-1]
                properties['itk_origin'] = sitk_image.GetOrigin()
                properties['itk_spacing'] = sitk_image.GetSpacing()
                properties['itk_direction'] = sitk_image.GetDirection()

            if 'png' in mod:
                # Scale to be [0-1]
                image_array /= 255
                # Greyscale image
                if len(image_array.shape) == 2:
                    all_ndarrays.append(image_array)
                else:
                    # Colour, append channels in R, G, B order.
                    for c in range(image_array.shape[-1]):
                        # Sitk reads png as [H, W, C], with channels in RGB order
                        all_ndarrays.append(image_array[:, :, c])
            else:
                # Medical images, assume single modality
                all_ndarrays.append(image_array)

        all_data = np.stack(all_ndarrays)

        if return_properties:
            properties['channel_intensity_properties'] = OrderedDict()
            for c in range(all_data.shape[0]):
                properties['channel_intensity_properties'][c] = OrderedDict()
                properties['channel_intensity_properties'][c]['mean'] = np.mean(all_data[c])
                properties['channel_intensity_properties'][c]['sd'] = np.std(all_data[c])

            return all_data, properties
        else:
            return all_data

    def resample_and_normalise(self, data, target_spacing, properties, force_separate_z):
        original_spacing_transposed = np.array(properties['original_spacing'])[self.transpose_forward]
        before = {
            'spacing': properties['original_spacing'],
            'spacing_transposed': original_spacing_transposed,
            'data.shape (transposed)': data.shape
        }

        data[np.isnan(data)] = 0

        data = resample_patient(data, original_spacing_transposed, target_spacing, 3, force_separate_z=force_separate_z,
                                order_z=0,
                                separate_z_anisotropy_threshold=self.resample_separate_z_anisotropy_threshold)

        after = {
            'spacing': target_spacing,
            'new data.shape': data.shape
        }
        print('Before: ', before, '\nAfter: ', after, '\n')

        properties['size_after_resampling'] = data[0].shape
        properties['spacing_after_resampling'] = target_spacing

        return normalise(data, self.normalisation_scheme_per_modality, self.intensity_properties,
                         properties['channel_intensity_properties']), properties

    def _run_internal(self, target_spacing, sample_identifier, sample_properties, output_folder_stage: Path,
                      input_folder, force_separate_z):
        sample_data = self.load_and_combine(sample_identifier, input_folder)

        sample_data = sample_data.transpose((0, *[i + 1 for i in self.transpose_forward]))

        resampled_data, sample_properties = self.resample_and_normalise(sample_data, target_spacing, sample_properties,
                                                                        force_separate_z)

        output_file_path = output_folder_stage / f'{sample_identifier}.npz'
        output_properties_path = output_folder_stage / f'{sample_identifier}.pkl'

        # Pass empty array as default, as np.save/load converts None to array of None, which isn't nice
        fg_mask = get_object_mask(resampled_data) if self.make_foreground_masks else make_default_mask()

        print('Saving ', output_file_path)
        np.savez_compressed(output_file_path, data=resampled_data, mask=fg_mask)
        save_pickle(sample_properties, output_properties_path)

    def run(self, target_spacings, input_folder: Path, sample_identifiers, sample_properties: OrderedDict,
            output_folder: Path, data_identifier, num_proc=default_num_processes, force_separate_z=None):

        print('Initialising to run preprocessing')
        print('Input folder: ', input_folder)
        print('Output folder: ', output_folder)

        num_stages = len(target_spacings)
        if not isinstance(num_proc, (list, tuple, np.ndarray)):
            num_proc = [num_proc] * num_stages

        assert len(num_proc) == num_stages

        for i in range(num_stages):
            output_folder_stage = output_folder / f'{data_identifier}_stage{i}'
            output_folder_stage.mkdir(parents=True, exist_ok=True)

            spacing = target_spacings[i]
            all_args = map(lambda s_i: [spacing, s_i, sample_properties[s_i], output_folder_stage, input_folder,
                                        force_separate_z],
                           sample_identifiers)

            with Pool(num_proc[i]) as pool:
                pool.starmap(self._run_internal, all_args)

    def preprocess_test_case(self, target_spacing, input_folder: Path, test_sample_identifier: str,
                             force_separate_z=None):
        test_sample_data, properties = self.load_and_combine(test_sample_identifier, input_folder,
                                                             return_properties=True)

        test_sample_data = test_sample_data.transpose((0, *[i + 1 for i in self.transpose_forward]))
        resampled_data, properties = self.resample_and_normalise(test_sample_data, target_spacing, properties,
                                                                 force_separate_z)

        return resampled_data.astype(np.float32), properties
