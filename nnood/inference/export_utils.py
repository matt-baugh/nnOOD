import os
from copy import deepcopy
from pathlib import Path
from typing import Union

import numpy as np
import SimpleITK as sitk

from nnood.preprocessing.preprocessing import get_lowres_axis, get_do_separate_z, resample_data
from nnood.utils.file_operations import save_pickle


def save_data_as_file(data: Union[str, Path, np.ndarray], out_file_name: Path,
                      properties_dict: dict, order: int = 1,
                      postprocess_fn: callable = None, postprocess_args: tuple = None,
                      resampled_npz_file_name: Path = None,
                      non_postprocessed_file_name: Path = None, force_separate_z: bool = None,
                      interpolation_order_z: int = 0, verbose: bool = True):
    """
    This is a utility for writing data to nifti/png and npz. It requires the data to have been preprocessed by
    GenericPreprocessor because it depends on the property dictionary output (dct) to know the geometry of the original
    data. data does not have to have the same size in pixels as the original data, it will be
    resampled to match that. This is generally useful because the spacings our networks operate on are most of the time
    not the native spacings of the image data.
    If postprogess_fn is not None then postprogess_fn(data, *postprocess_args)
    will be called before nifti export
    There is a problem with python process communication that prevents us from communicating obejcts
    larger than 2 GB between processes (basically when the length of the pickle string that will be sent is
    communicated by the multiprocessing.Pipe object then the placeholder (I think) does not allow for long
    enough strings (lol). This could be fixed by changing i to l (for long) but that would require manually
    patching system python code.) We circumvent that problem here by saving pred to a npy file that will
    then be read (and finally deleted) by the Process. save_score_as_nifti can take either
    filename or np.ndarray for data and will handle this automatically
    :param data: Image with shape (c, [z, ], y, x) - channels first.
    :param out_file_name:
    :param properties_dict:
    :param order:
    :param postprocess_fn:
    :param postprocess_args:
    :param resampled_npz_file_name:
    :param non_postprocessed_file_name:
    :param force_separate_z: if None then we dynamically decide how to resample along z, if True/False then always
    /never resample along z separately. Do not touch unless you know what you are doing
    :param interpolation_order_z: if separate z resampling is done then this is the order for resampling in z
    :param verbose:
    :return:
    """
    if verbose:
        print("force_separate_z:", force_separate_z, "interpolation order:", order)

    if isinstance(data, str) or isinstance(data, Path):
        data = Path(data)
        assert data.is_file()
        del_file = deepcopy(data)
        data = np.load(data)
        os.remove(del_file)

    # first resample, then put result into bbox of cropping, then save
    current_shape = data.shape
    shape_original = properties_dict.get('original_size')

    if np.any([i != j for i, j in zip(np.array(current_shape[1:]), np.array(shape_original))]):
        if force_separate_z is None:
            if get_do_separate_z(properties_dict.get('original_spacing')):
                do_separate_z = True
                lowres_axis = get_lowres_axis(properties_dict.get('original_spacing'))
            elif get_do_separate_z(properties_dict.get('spacing_after_resampling')):
                do_separate_z = True
                lowres_axis = get_lowres_axis(properties_dict.get('spacing_after_resampling'))
            else:
                do_separate_z = False
                lowres_axis = None
        else:
            do_separate_z = force_separate_z
            if do_separate_z:
                lowres_axis = get_lowres_axis(properties_dict.get('original_spacing'))
            else:
                lowres_axis = None

        if lowres_axis is not None and len(lowres_axis) != 1:
            # this happens for spacings like (0.24, 1.25, 1.25) for example. In that case we do not want to resample
            # separately in the out of plane axis
            do_separate_z = False

        if verbose:
            print("separate z:", do_separate_z, "lowres axis", lowres_axis)
        data_old_spacing = resample_data(data, shape_original, axis=lowres_axis, order=order,
                                         do_separate_z=do_separate_z, order_z=interpolation_order_z)
    else:
        if verbose:
            print("no resampling necessary")
        data_old_spacing = data

    if resampled_npz_file_name is not None:
        np.savez_compressed(resampled_npz_file_name, data=data_old_spacing)
        save_pickle(properties_dict, resampled_npz_file_name.with_suffix('.pkl'))

    # Currently we don't have a separate cropping stage, due to the lack of map to say where background is
    # I'll leave this logic commented in case we want it back later

    # bbox = properties_dict.get('crop_bbox')
    # if bbox is not None:
    #     data_old_size = np.zeros(shape_original_before_cropping)
    #     for c in range(3):
    #         bbox[c][1] = np.min((bbox[c][0] + data_old_spacing.shape[c], shape_original_before_cropping[c]))
    #     data_old_size[bbox[0][0]:bbox[0][1],
    #     bbox[1][0]:bbox[1][1],
    #     bbox[2][0]:bbox[2][1]] = data_old_spacing
    # else:
    #     data_old_size = data_old_spacing

    if postprocess_fn is not None:
        data_old_size_postprocessed = postprocess_fn(np.copy(data_old_spacing), *postprocess_args)
    else:
        data_old_size_postprocessed = data_old_spacing

    def _save_array(data_to_save: np.ndarray, file_path: Path):
        # Move to channels last, to match SITK representation
        data_to_save = np.moveaxis(data_to_save, 0, -1)

        if file_path.suffix == '.png':
            # To be saved as a png the image must be uint8, with values in range [0,255]
            data_to_save = (data_to_save * 255).astype(np.uint8)

        data_to_save_itk = sitk.GetImageFromArray(data_to_save, isVector=True)
        data_to_save_itk.SetSpacing(properties_dict['itk_spacing'])
        data_to_save_itk.SetOrigin(properties_dict['itk_origin'])
        data_to_save_itk.SetDirection(properties_dict['itk_direction'])
        sitk.WriteImage(data_to_save_itk, file_path.__str__())

    _save_array(data_old_size_postprocessed, out_file_name)

    if (non_postprocessed_file_name is not None) and (postprocess_fn is not None):
        _save_array(data_old_spacing, non_postprocessed_file_name)
