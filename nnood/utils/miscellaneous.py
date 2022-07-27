from collections import OrderedDict
import importlib
import pkgutil
from pathlib import Path
import re
from typing import Dict, List, Tuple

import numpy as np
import torch


class no_op:
    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass


def recursive_find_python_class(folder, class_name, current_module):
    tr = None
    for importer, modname, ispkg in pkgutil.iter_modules(folder):
        if not ispkg:
            m = importlib.import_module(current_module + "." + modname)
            if hasattr(m, class_name):
                tr = getattr(m, class_name)
                break

    if tr is None:
        for importer, modname, ispkg in pkgutil.iter_modules(folder):
            if ispkg:
                next_current_module = current_module + "." + modname
                tr = recursive_find_python_class([folder[0], modname], class_name,
                                                 current_module=next_current_module)
            if tr is not None:
                break

    return tr


def load_pretrained_weights(network, fname, verbose=False):
    """
    THIS DOES NOT TRANSFER SEGMENTATION HEADS!
    """
    saved_model = torch.load(fname)
    pretrained_dict = saved_model['state_dict']

    new_state_dict = {}

    # if state dict comes form nn.DataParallel but we use non-parallel model here then the state dict keys do not
    # match. Use heuristic to make it match
    for k, value in pretrained_dict.items():
        key = k
        # remove module. prefix from DDP models
        if key.startswith('module.'):
            key = key[7:]
        new_state_dict[key] = value

    pretrained_dict = new_state_dict

    model_dict = network.state_dict()
    ok = True
    for key, _ in model_dict.items():
        if 'conv_blocks' in key:
            if (key in pretrained_dict) and (model_dict[key].shape == pretrained_dict[key].shape):
                continue
            else:
                ok = False
                break

    # filter unnecessary keys
    if ok:
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                           (k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape)}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        print("################### Loading pretrained weights from file ", fname, '###################')
        if verbose:
            print("Below is the list of overlapping blocks in pretrained model and nnUNet architecture:")
            for key, _ in pretrained_dict.items():
                print(key)
        print("################### Done ###################")
        network.load_state_dict(model_dict)
    else:
        raise RuntimeError("Pretrained weights are not compatible with the current network architecture")


def get_sample_ids_and_files(input_folder: Path, expected_modalities: Dict[int, str]) -> List[Tuple[str, List[Path]]]:

    assert input_folder.is_dir()

    id_to_files_dict = OrderedDict()

    for f in input_folder.iterdir():
        if not f.is_file():
            continue

        match = re.fullmatch('(.*)_(\d\d\d\d)(\..*)', f.name)

        if match is None:
            print(f'File "{f}" does not match the expected name format, so will be ignored')
            continue

        sample_id, mod_num, suffix = match.groups()
        mod_num = int(mod_num)

        if sample_id not in id_to_files_dict:
            id_to_files_dict[sample_id] = OrderedDict()

        id_to_files_dict[sample_id][mod_num] = f

    invalid_folder = False
    for sample_id, files in id_to_files_dict.items():

        missing_mods = [m for m in range(len(expected_modalities)) if m not in files]

        # noinspection PySimplifyBooleanCheck
        if missing_mods != []:  # Don't like simplification, removes clarity
            print(f'Sample "{sample_id}" is missing modalities: {missing_mods}')
            invalid_folder = True
            continue

        # Check modality formats
        for i, mod in expected_modalities.items():
            if 'png' in mod and files[i].suffix != '.png':
                print(f'Sample "{sample_id}" expected a .png for modality {i}: {files[i]}')
                invalid_folder = True

    if invalid_folder:
        raise RuntimeError(f'Problems with files in {input_folder}')

    id_to_files_list = []
    # convert dictionary to list, return
    for sample_id, file_dict in id_to_files_dict.items():
        sample_files = list(file_dict.values())
        sample_files.sort()

        id_to_files_list.append((sample_id, sample_files))

    return id_to_files_list


def make_pos_enc(sample_shape: np.ndarray) -> np.ndarray:
    """
    :param sample_shape: Shape of sample data, excluding channels dimension
    :return: positional encoding of shape
    """
    # One coordinate encoding channel per dimension
    sample_coords = np.zeros((len(sample_shape), *sample_shape))

    for dim_num, dim_size in enumerate(sample_shape):
        # Coordinates range from -1 to 1 in each dimension
        dim_coords = np.linspace(-1, 1, dim_size)

        # Expand coords so they are correctly broadcast (meaning they only change in their respective dimension)
        for _ in range(dim_num):
            dim_coords = np.expand_dims(dim_coords, axis=0)

        for _ in range(len(sample_shape) - dim_num - 1):
            dim_coords = np.expand_dims(dim_coords, axis=-1)

        sample_coords[dim_num] = dim_coords

    return sample_coords


def make_hypersphere_mask(radius: int, dims: int):
    L = np.arange(-radius, radius + 1)
    # It thinks meshgrid returns a string for some reason
    # noinspection PyTypeChecker
    mg: List[np.ndarray] = np.meshgrid(*([L] * dims))
    return np.sum([D ** 2 for D in mg], axis=0) <= radius ** 2


def make_default_mask():
    return np.array([])
