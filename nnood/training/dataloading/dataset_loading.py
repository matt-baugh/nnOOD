from collections import OrderedDict
from multiprocessing import Pool
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np

from nnood.configuration import default_num_processes
from nnood.network_architecture.neural_network import AnomalyScoreNetwork
from nnood.self_supervised_task.self_sup_task import SelfSupTask
from nnood.utils.file_operations import load_pickle
from nnood.utils.miscellaneous import make_pos_enc


def load_dataset_filenames(folder, sample_identifiers, num_cases_properties_loading_threshold=1000):
    # we don't load the actual data but instead return the filename to the np file.
    sample_identifiers.sort()
    dataset = OrderedDict()
    for s_i in sample_identifiers:
        dataset[s_i] = OrderedDict()
        dataset[s_i]['data_file'] = folder / f'{s_i}.npz'
        dataset[s_i]['properties_file'] = folder / f'{s_i}.pkl'

    if len(sample_identifiers) <= num_cases_properties_loading_threshold:
        print('Loading all case properties')
        for i in dataset.keys():
            dataset[i]['properties'] = load_pickle(dataset[i]['properties_file'])

    return dataset


# Temporary, can be replaced with Path.with_stem in 3.9, but scared to update code atm
def path_with_stem(p: Path, new_stem: str):
    return p.with_name(new_stem + p.suffix)


def convert_to_npy(args):
    if not isinstance(args, tuple):
        keys = ['data']
        npz_file: Path = args
    else:
        npz_file, keys = args

    npy_file = npz_file.with_suffix('.npy')

    unpack_files = not npy_file.is_file()

    if not unpack_files:
        for k in keys[1:]:
            mask_file = path_with_stem(npy_file, npy_file.stem + f'_{k}')
            unpack_files = not mask_file.is_file()

            # As soon as we find one missing, break loop
            if unpack_files:
                break

    if unpack_files:
        data_dict = np.load(npz_file)

        # Assume first key is main one
        np.save(npy_file, data_dict[keys[0]])

        # Save others with main stem plus key
        for k in keys[1:]:
            np.save(path_with_stem(npy_file, npy_file.stem + f'_{k}'), data_dict[k])


def unpack_dataset(folder: Path, sample_identifiers, threads=default_num_processes, keys=['data', 'mask']):
    """
    unpacks all npz files in a folder to npy (whatever you want to have unpacked must be saved under key)
    :param folder:
    :param sample_identifiers
    :param threads:
    :param keys:
    :return:
    """
    p = Pool(threads)
    npz_files = [folder / f'{s_i}.npz' for s_i in sample_identifiers]
    p.map(convert_to_npy, zip(npz_files, [keys] * len(npz_files)))
    p.close()
    p.join()


# Cases are stored as npz, but we require unpack_dataset to be run. This will decompress them into npy which is much
# faster to access
def load_npy_or_npz(file: Path, npy_mmap_mode: str, load_mask: bool = False) \
        -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    npy_file = file.with_suffix('.npy')

    if npy_file.is_file():
        if load_mask:
            return np.load(npy_file, npy_mmap_mode), np.load(path_with_stem(npy_file, npy_file.stem + '_mask'))
        else:
            return np.load(npy_file, npy_mmap_mode)
    else:
        file_dict = np.load(file)
        if load_mask:
            return file_dict['data'], file_dict['mask']
        else:
            return file_dict['data']


def sample_lb_around_step_locations(steps: List[List[int]], patch_size: np.ndarray, dimension_lbs: np.ndarray,
                                    dimension_ubs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    target_lb_mean = np.array([np.random.choice(s) for s in steps])

    target_lb = target_lb_mean
    # target_lb = np.random.normal(target_lb_mean, patch_size / 8).round().astype(int)
    #
    # # lbs and ubs are INCLUSIVE
    # too_low = target_lb < dimension_lbs
    # too_high = target_lb > dimension_ubs
    #
    # while np.logical_or(too_low, too_high).any():
    #     # Reflect, but translate out 1 so 0 still has highest pdf
    #     target_lb[too_low] = (dimension_lbs + ((dimension_lbs - 1) - target_lb))[too_low]
    #     target_lb[too_high] = (dimension_ubs - (target_lb - (dimension_ubs + 1)))[too_high]
    #
    #     too_low = target_lb < dimension_lbs
    #     too_high = target_lb > dimension_ubs

    return target_lb, target_lb + patch_size


class DataLoader:
    def __init__(self, data, patch_size: np.ndarray, final_patch_size: np.ndarray, task: SelfSupTask, batch_size,
                 has_prev_stage=False, oversample_foreground_percent=0.0, load_dataset_ram=False,
                 data_has_foreground=False, memmap_mode='r', pad_mode='reflect', pad_kwargs_data=None, pad_sides=None):
        """
        This is a combination of batchgenerator's SlimDataLoaderBase and a generalised nnUNet DataLoader3D, to avoid
        dependency issues later down the line.
        """
        self._data = data
        self.task = task
        self.batch_size = batch_size
        self.thread_id = 0
        if pad_kwargs_data is None:
            pad_kwargs_data = OrderedDict()
        self.pad_kwargs_data = pad_kwargs_data
        self.pad_mode = pad_mode
        self.oversample_foreground_percent = oversample_foreground_percent
        self.load_dataset_ram = load_dataset_ram
        self.data_has_foreground = data_has_foreground
        self.final_patch_size = final_patch_size
        self.has_prev_stage = has_prev_stage
        self.patch_size = patch_size
        self.list_of_keys = list(self._data.keys())
        # need_to_pad denotes by how much we need to pad the data so that if we sample a patch of size final_patch_size
        # (which is what the network will get) these patches will also cover the border of the patients
        self.need_to_pad = (np.array(patch_size) - np.array(final_patch_size)).astype(int)
        if pad_sides is not None:
            if not isinstance(pad_sides, np.ndarray):
                pad_sides = np.array(pad_sides)
            self.need_to_pad += pad_sides
        self.memmap_mode = memmap_mode
        self.num_channels = None
        self.pad_sides = pad_sides
        self.output_data_shape, self.output_coords_shape, self.label_shape = self.determine_shapes()

        if self.load_dataset_ram:
            self._data_ram = OrderedDict()
            for s_i in self.list_of_keys:
                self._data_ram[s_i] = OrderedDict()
                data = load_npy_or_npz(self._data[s_i]['data_file'], self.memmap_mode, self.data_has_foreground)

                if self.data_has_foreground:
                    self._data_ram[s_i]['data'] = data[0]
                    self._data_ram[s_i]['mask'] = data[1]
                else:
                    self._data_ram[s_i]['data'] = data

                if 'properties' in self._data[s_i].keys():
                    self._data_ram[s_i]['properties'] = self._data[s_i]['properties']
                else:
                    self._data_ram[s_i]['properties'] = load_pickle(self._data[s_i]['properties_file'])
        else:
            self._data_ram = None

    def get_do_oversample(self, batch_idx):
        return not batch_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))

    def determine_shapes(self):
        if self.has_prev_stage:
            num_label = 2
        else:
            num_label = 1

        k = list(self._data.keys())[0]
        case_all_data = load_npy_or_npz(self._data[k]['data_file'], self.memmap_mode)

        num_image_channels = case_all_data.shape[0]
        num_patch_dimensions = len(self.patch_size)
        data_shape = (self.batch_size, num_image_channels, *self.patch_size)
        coords_shape = (self.batch_size, num_patch_dimensions, *self.patch_size)
        label_shape = (self.batch_size, num_label, *self.patch_size)
        return data_shape, coords_shape, label_shape

    def get_random_sample(self, load_mask: bool):
        key = np.random.choice(self.list_of_keys)
        if self.load_dataset_ram:
            if load_mask:
                return (self._data_ram[key]['data'], self._data_ram[key]['mask']), self._data_ram[key]['properties']
            else:
                return self._data_ram[key]['data'], self._data_ram[key]['properties']
        else:
            if 'properties' in self._data[key].keys():
                return load_npy_or_npz(self._data[key]['data_file'], self.memmap_mode, load_mask), \
                       self._data[key]['properties']
            else:
                return load_npy_or_npz(self._data[key]['data_file'], self.memmap_mode, load_mask), \
                       load_pickle(self._data[key]['properties_file'])

    def generate_train_batch(self):
        selected_keys = np.random.choice(self.list_of_keys, self.batch_size, True, None)
        data = np.zeros(self.output_data_shape, dtype=np.float32)
        coords = np.zeros(self.output_coords_shape, dtype=np.float32)
        scores = np.zeros(self.label_shape, dtype=np.float32)
        case_properties = []
        for batch_index, data_key in enumerate(selected_keys):
            if self.load_dataset_ram:
                sample_data = self._data_ram[data_key]['data']
                sample_mask = self._data_ram[data_key]['mask'] if self.data_has_foreground else None
                properties = self._data_ram[data_key]['properties']
            else:
                if 'properties' in self._data[data_key].keys():
                    properties = self._data[data_key]['properties']
                else:
                    properties = load_pickle(self._data[data_key]['properties_file'])

                sample_data = load_npy_or_npz(self._data[data_key]['data_file'], self.memmap_mode,
                                              self.data_has_foreground)

                if self.data_has_foreground:
                    sample_mask = sample_data[1]
                    sample_data = sample_data[0]
                else:
                    sample_mask = None

            case_properties.append(properties)

            # Assume label has shape [1,[ z,] y, x]
            # Also assume task does not edit sample_data in place
            sample_data, sample_label, anomaly_centres = self.task(sample_data, sample_mask, properties,
                                                                   self.get_random_sample, return_locations=True)

            # Construct positional encoding
            original_sample_shape = sample_data.shape[1:]

            sample_coords = make_pos_enc(original_sample_shape)

            need_to_pad = self.need_to_pad.copy()
            for pad_dim in range(len(self.patch_size)):
                # if sample_data.shape + need_to_pad is still < patch size we need to pad more!
                if need_to_pad[pad_dim] + sample_data.shape[pad_dim + 1] < self.patch_size[pad_dim]:
                    need_to_pad[pad_dim] = self.patch_size[pad_dim] - sample_data.shape[pad_dim + 1]

            # we can now choose the bbox from -need_to_pad // 2 to shape - patch_size + need_to_pad // 2. Here we
            # define what the upper and lower bound can be to then sample form them with np.random.randint
            dimension_lbs = np.zeros(len(original_sample_shape), dtype=np.int)
            dimension_ubs = np.zeros(len(original_sample_shape), dtype=np.int)
            for dim in range(len(original_sample_shape)):
                dimension_lbs[dim] = -need_to_pad[dim] // 2
                dimension_ubs[dim] = original_sample_shape[dim] + need_to_pad[dim] // 2 + need_to_pad[dim] % 2 - \
                                     self.patch_size[dim]

            force_fg = self.get_do_oversample(batch_index)

            # Get inference times step locations for this sample. Use final patch size to get correct locations (larger
            # patch_size could make one step be missed).
            inference_steps = AnomalyScoreNetwork.compute_steps_for_sliding_window(tuple(self.final_patch_size),
                                                                                   original_sample_shape,
                                                                                   0.5)

            bbox_lbs, bbox_ubs = sample_lb_around_step_locations(inference_steps, self.patch_size, dimension_lbs,
                                                                 dimension_ubs)

            while force_fg and len(anomaly_centres) != 0 and \
                    not any([(bbox_lbs <= a_c).all() and (a_c <= bbox_ubs).all() for a_c in anomaly_centres]):
                bbox_lbs, bbox_ubs = sample_lb_around_step_locations(inference_steps, self.patch_size, dimension_lbs,
                                                                     dimension_ubs)

            # We first crop the data to the region of the bbox that actually lies within the data. This will result in a
            # smaller array which is then faster to pad. valid_bbox is just the coord that lied within the data cube. It
            # will be padded to match the patch size later
            valid_bbox_lbs = np.maximum(bbox_lbs, 0)
            valid_bbox_ubs = np.minimum(bbox_ubs, original_sample_shape)
            valid_bbox_slices = tuple([slice(None), *[slice(lb, ub) for lb, ub in zip(valid_bbox_lbs, valid_bbox_ubs)]])

            # TODO: Revisit comment, should I use a regression mode????????
            # If we are doing the cascade then we will also need to load the score of the previous stage and
            # concatenate it. Here it will be concatenates to the current label because the augmentations need to be
            # applied to it in segmentation mode. Later in the data augmentation we move it from the segmentations to
            # the last channel of the data
            if self.has_prev_stage:
                assert False, 'You\'e tried to use a cascade dataloader, but I haven\'t made it yet!!!!!!!!'
                score_from_previous_stage = None
                assert all([i == j for i, j in zip(score_from_previous_stage.shape[1:], sample_data.shape[1:])]), \
                    'score_from_previous_stage does not match the shape of sample_data: %s vs %s' % \
                    (str(score_from_previous_stage.shape[1:]), str(sample_data.shape[1:]))
            else:
                score_from_previous_stage = None

            # TODO: Review whether we still want this
            # At this point you might ask yourself why we would treat curr_label differently from
            # score_from_previous_stage. Why not just concatenate them here and forget about the if statements? Well
            # that's because curr_label needs to be padded with -1 constant whereas score_from_previous_stage needs to
            # be padded with 0s (we could also  remove label -1 in the data augmentation but this way it is less error
            # prone)
            # -- Attempt pad both with 0

            sample_data = np.copy(sample_data[valid_bbox_slices])
            sample_label = np.copy(sample_label[valid_bbox_slices])
            sample_coords = np.copy(sample_coords[valid_bbox_slices])
            if score_from_previous_stage is not None:
                score_from_previous_stage = score_from_previous_stage[valid_bbox_slices]

            padding = ((0, 0), *[(-min(0, lb), max(ub - s, 0)) for lb, ub, s in zip(bbox_lbs, bbox_ubs,
                                                                                    original_sample_shape)])

            data[batch_index] = np.pad(sample_data, padding, self.pad_mode, **self.pad_kwargs_data)
            # reflect_type = 'odd' continues coordinate steps outside of -1, 1 region
            coords[batch_index] = np.pad(sample_coords, padding, 'reflect', reflect_type='odd')
            scores[batch_index, 0] = np.pad(sample_label, padding, self.pad_mode, **self.pad_kwargs_data)
            if score_from_previous_stage is not None:
                scores[batch_index, 1] = np.pad(score_from_previous_stage, padding, self.pad_mode,
                                                **self.pad_kwargs_data)

        return {'data': data, 'coords': coords, 'seg': scores, 'properties': case_properties, 'keys': selected_keys}

    def set_thread_id(self, thread_id):
        self.thread_id = thread_id

    def __iter__(self):
        return self

    def __next__(self):
        return self.generate_train_batch()
