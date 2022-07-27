from collections import OrderedDict
from multiprocessing import Pool
from typing import List, Dict, Optional
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from nnood.preprocessing.normalisation import GLOBAL_NORMALISATION_MODALITIES
from nnood.utils.file_operations import load_json, save_pickle
from nnood.paths import DATASET_JSON_FILE, DATASET_PROPERTIES_FILE


class DatasetAnalyser:
    def __init__(self, raw_data_path: Path, num_processes: int):

        self.raw_data_path = raw_data_path
        self.num_processes = num_processes
        self.sizes = None
        self.spacings = None

        self.dataset_json = load_json(raw_data_path / DATASET_JSON_FILE)

        self.sample_identifiers: List[str] = self.dataset_json['training']

    def get_modalities(self):
        str_modalities = self.dataset_json['modality']
        return {int(k): str_modalities[k] for k in str_modalities}

    @staticmethod
    def requires_global_normalisation(modality: str):
        return modality.lower() in GLOBAL_NORMALISATION_MODALITIES

    @staticmethod
    def get_voxels_in_foreground(image_array: np.ndarray) -> np.ndarray:

        assert len(image_array.shape) in [2, 3], f'Image must have 2 or 3 dimensions: {len(image_array.shape)}'

        mask = image_array > 0
        # Only include every 5, reducing memory with little effect on global statistics
        return image_array[mask][::5]

    def analyse_sample(self, sample_id: str) -> (OrderedDict, Dict[str, Optional[np.ndarray]]):

        modalities = self.get_modalities()
        properties = OrderedDict()
        properties['sample_id'] = sample_id
        properties['data_files'] = []
        channel_intensity_properties = []
        modality_intensities = OrderedDict()

        for i in range(len(modalities)):
            mod = modalities[i]
            suffix = 'png' if 'png' in mod else 'nii.gz'
            file_path = self.raw_data_path / 'imagesTr' / f'{sample_id}_{i:04d}.{suffix}'

            properties['data_files'].append(file_path)

            # Assume the size/spacing statistics are equal across modalities of sample
            itk_image = sitk.ReadImage(file_path.__str__())
            if i == 0:
                properties['original_size'] = np.array(itk_image.GetSize())[::-1]
                properties['original_spacing'] = np.array(itk_image.GetSpacing())[::-1]
                properties['itk_origin'] = itk_image.GetOrigin()
                properties['itk_spacing'] = itk_image.GetSpacing()
                properties['itk_direction'] = itk_image.GetDirection()

            image_array = sitk.GetArrayFromImage(itk_image).astype(np.float32)

            if 'png' in mod:
                image_array /= 255

            if DatasetAnalyser.requires_global_normalisation(mod):
                modality_intensities[mod] = DatasetAnalyser.get_voxels_in_foreground(image_array)
            else:
                modality_intensities[mod] = None

            if 'png' in mod and len(image_array.shape) != 2:
                for c in range(image_array.shape[-1]):
                    d = OrderedDict()
                    curr_channel = image_array[:, :, c]
                    d['mean'] = np.mean(curr_channel)
                    d['sd'] = np.std(curr_channel)
                    channel_intensity_properties.append(d)
            else:
                d = OrderedDict()
                d['mean'] = np.mean(image_array)
                d['sd'] = np.std(image_array)
                channel_intensity_properties.append(d)

        properties['channel_intensity_properties'] = OrderedDict()
        for c in range(len(channel_intensity_properties)):
            properties['channel_intensity_properties'][c] = channel_intensity_properties[c]

        return properties, modality_intensities

    def calc_intensity_properties(self, all_sample_intensities: List[Dict[str, Optional[np.ndarray]]])\
            -> Dict[str, Optional[OrderedDict]]:

        modality_statistics = OrderedDict()

        mods = self.get_modalities()
        for i in range(len(mods)):
            mod = mods[i]
            if all_sample_intensities[0][mod] is None:
                modality_statistics[mod] = None
            else:
                all_mod_intensities = np.concatenate(list(map(lambda d: d[mod], all_sample_intensities)))

                modality_statistics[i] = OrderedDict()
                modality_statistics[i]['median'] = np.median(all_mod_intensities)
                modality_statistics[i]['mean'] = np.mean(all_mod_intensities)
                modality_statistics[i]['sd'] = np.std(all_mod_intensities)
                modality_statistics[i]['min'] = np.min(all_mod_intensities)
                modality_statistics[i]['max'] = np.max(all_mod_intensities)
                modality_statistics[i]['percentile_99_5'] = np.percentile(all_mod_intensities, 99.5)
                modality_statistics[i]['percentile_00_5'] = np.percentile(all_mod_intensities, 00.5)

        return modality_statistics

    def analyse_dataset(self):

        with Pool(self.num_processes) as pool:
            all_sample_analytics = pool.map(self.analyse_sample, self.sample_identifiers)

        # List of tuples to tuple of lists
        all_sample_properties, all_sample_intensities = map(list, zip(*all_sample_analytics))

        dataset_properties = OrderedDict()
        dataset_properties['all_sizes'] = list(map(lambda ps: ps['original_size'], all_sample_properties))
        dataset_properties['all_spacings'] = list(map(lambda ps: ps['original_spacing'], all_sample_properties))
        # dataset_properties['all_data_files'] = map(lambda ps: ps['data_files'], all_sample_properties)
        dataset_properties['sample_identifiers'] = self.sample_identifiers
        dataset_properties['modalities'] = self.get_modalities()
        dataset_properties['intensity_properties'] = self.calc_intensity_properties(all_sample_intensities)
        dataset_properties['tensor_dimensions'] = self.dataset_json['tensorImageSize']
        dataset_properties['data_augs'] = self.dataset_json['data_augs']
        dataset_properties['has_uniform_background'] = self.dataset_json['has_uniform_background']

        dataset_properties['sample_properties'] = OrderedDict()
        for sample_id in self.sample_identifiers:
            dataset_properties['sample_properties'][sample_id] = next(s_p for s_p in all_sample_properties
                                                                      if s_p['sample_id'] == sample_id)

        save_pickle(dataset_properties, self.raw_data_path / DATASET_PROPERTIES_FILE)
        return dataset_properties
