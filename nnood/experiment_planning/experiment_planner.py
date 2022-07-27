from pathlib import Path
from collections import OrderedDict
from copy import deepcopy

import numpy as np

from nnood.utils.file_operations import load_pickle, save_pickle
from nnood.paths import DATASET_PROPERTIES_FILE, default_plans_identifier, default_data_identifier
from nnood.experiment_planning.utils import get_pool_and_conv_props
from nnood.experiment_planning.modality_conversion import num_modality_components, get_channel_list
from nnood.preprocessing.normalisation import modality_norm_scheme
from nnood.network_architecture.generic_UNet import Generic_UNet
from nnood.preprocessing.preprocessing import GenericPreprocessor
from nnood.configuration import default_num_processes


# Based on ExperimentPlanner3D_v21 from original nnU-Net
class ExperimentPlanner:

    def __init__(self, raw_data_path: Path, preprocessed_data_path: Path, num_processes: int, num_disable_skip: int):
        self.raw_data_path = raw_data_path
        self.preprocessed_data_path = preprocessed_data_path
        self.num_processes = num_processes
        self.num_disable_skip = num_disable_skip

        self.dataset_properties = load_pickle(preprocessed_data_path / DATASET_PROPERTIES_FILE)

        self.plans_per_stage = OrderedDict()
        self.plans = OrderedDict()
        self.plans_path = self.preprocessed_data_path / default_plans_identifier
        self.data_identifier = default_data_identifier

        # Should be '4d' or '3d', includes 'channel' dimension
        self.tensor_dimensions = self.dataset_properties['tensor_dimensions'].lower()

        # Set during experiment planning
        self.transpose_forward = None
        self.transpose_backward = None

        # Network parameter boundaries
        self.unet_base_num_features = Generic_UNet.BASE_NUM_FEATURES
        self.unet_max_num_filters = Generic_UNet.MAX_NUM_FILTERS_3D if self.using_3d_data() \
            else Generic_UNet.MAX_FILTERS_2D
        self.unet_max_num_pool = Generic_UNet.MAX_NUM_POOL
        self.unet_min_batch_size = Generic_UNet.DEFAULT_BATCH_SIZE
        self.unet_feature_map_min_edge_length = Generic_UNet.MIN_FEATURE_EDGE_LEN

        self.target_spacing_percentile = 50
        self.anisotropy_threshold = 3
        self.proportion_of_patient_visible_at_stage_0 = 1 / 4
        self.max_proportional_batch_size = 0.05  # Batch size is at most 5% of dataset
        self.conv_per_stage = 2

    def using_3d_data(self):
        return self.tensor_dimensions == '4d'

    def get_properties_for_stage(self, current_spacing: np.ndarray, original_spacing: np.ndarray,
                                 original_shape: np.ndarray, num_cases: int, num_modalities: int):
        # Network also takes positional encoding, one channel for each dimension
        num_input_channels = num_modalities + len(original_shape)

        # Size of full-res median shape when using current spacing
        new_median_shape = np.round(original_spacing / current_spacing * original_shape).astype(int)
        dataset_num_voxels = np.prod(new_median_shape) * num_cases

        # Compute how many voxels are one mm
        input_patch_size = 1 / np.array(current_spacing)

        # Normalize voxels per mm
        input_patch_size /= input_patch_size.mean()

        # create an isotropic patch of size 512x512x512mm
        input_patch_size *= 1 / min(input_patch_size) * 512  # to get a starting value
        input_patch_size = np.round(input_patch_size).astype(int)

        # clip it to the median shape of the dataset because patches larger then that make not much sense
        input_patch_size = np.array([min(i, j) for i, j in zip(input_patch_size, new_median_shape)])

        network_num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, new_shape, shape_must_be_divisible_by \
            = get_pool_and_conv_props(input_patch_size,
                                      self.unet_feature_map_min_edge_length,
                                      self.unet_max_num_pool,
                                      current_spacing)

        # Unsure of 2D case, revisit numbers if gives weirdly small/big batch sizes (target 50?)
        ref = Generic_UNet.use_this_for_batch_size_computation_3D if self.using_3d_data()\
            else Generic_UNet.use_this_for_batch_size_computation_2D
        here = Generic_UNet.compute_approx_vram_consumption(new_shape, network_num_pool_per_axis,
                                                            self.unet_base_num_features,
                                                            self.unet_max_num_filters, num_input_channels,
                                                            pool_op_kernel_sizes, conv_per_stage=self.conv_per_stage)
        while here > ref:
            axis_to_be_reduced = np.argsort(new_shape / new_median_shape)[-1]

            tmp = deepcopy(new_shape)
            tmp[axis_to_be_reduced] -= shape_must_be_divisible_by[axis_to_be_reduced]
            _, _, _, _, shape_must_be_divisible_by_new = \
                get_pool_and_conv_props(tmp,
                                        self.unet_feature_map_min_edge_length,
                                        self.unet_max_num_pool,
                                        current_spacing)
            new_shape[axis_to_be_reduced] -= shape_must_be_divisible_by_new[axis_to_be_reduced]

            # we have to recompute num_pool now:
            network_num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, new_shape, shape_must_be_divisible_by \
                = get_pool_and_conv_props(new_shape,
                                          self.unet_feature_map_min_edge_length,
                                          self.unet_max_num_pool,
                                          current_spacing)

            here = Generic_UNet.compute_approx_vram_consumption(new_shape, network_num_pool_per_axis,
                                                                self.unet_base_num_features,
                                                                self.unet_max_num_filters, num_input_channels,
                                                                pool_op_kernel_sizes,
                                                                conv_per_stage=self.conv_per_stage)

        input_patch_size = new_shape

        batch_size = Generic_UNet.DEFAULT_BATCH_SIZE
        batch_size = int(np.floor(max(ref / here, 1) * batch_size))

        # check if batch size is too large
        max_batch_size = np.round(self.max_proportional_batch_size * dataset_num_voxels /
                                  np.prod(input_patch_size, dtype=np.int64)).astype(int)
        max_batch_size = max(max_batch_size, self.unet_min_batch_size)
        batch_size = max(1, min(batch_size, max_batch_size))

        do_dummy_2D_data_aug = (max(input_patch_size) / input_patch_size[
            0]) > self.anisotropy_threshold

        plan = {
            'batch_size': batch_size,
            'num_pool_per_axis': network_num_pool_per_axis,
            'patch_size': input_patch_size,
            'median_patient_size_in_voxels': new_median_shape,
            'current_spacing': current_spacing,
            'original_spacing': original_spacing,
            'do_dummy_2D_data_aug': do_dummy_2D_data_aug,
            'pool_op_kernel_sizes': pool_op_kernel_sizes,
            'conv_kernel_sizes': conv_kernel_sizes,
        }
        return plan

    def save_my_plans(self):
        save_pickle(self.plans, self.plans_path)

    def load_my_plans(self):
        self.plans = load_pickle(self.plans_path)

        self.plans_per_stage = self.plans['plans_per_stage']
        self.dataset_properties = self.plans['dataset_properties']

        self.transpose_forward = self.plans['transpose_forward']
        self.transpose_backward = self.plans['transpose_backward']

    def plan_experiment(self):
        spacings = self.dataset_properties['all_spacings']
        sizes = self.dataset_properties['all_sizes']
        modalities = self.dataset_properties['modalities']
        num_modalities = sum([num_modality_components(m) for m in modalities.values()])

        target_spacing = np.percentile(np.vstack(spacings), self.target_spacing_percentile, 0)
        new_shapes = [np.array(i) / target_spacing * np.array(j) for i, j in zip(spacings, sizes)]

        max_spacing_axis = np.argmax(target_spacing)
        remaining_axes = [i for i in range(len(target_spacing)) if i != max_spacing_axis]
        self.transpose_forward = [max_spacing_axis] + remaining_axes
        self.transpose_backward = [np.argwhere(np.array(self.transpose_forward) == i)[0][0]
                                   for i in range(len(target_spacing))]

        median_shape = np.median(np.vstack(new_shapes), 0)
        max_shape = np.max(np.vstack(new_shapes), 0)
        min_shape = np.min(np.vstack(new_shapes), 0)
        print('Median shape: ', median_shape)
        print('Max shape: ', max_shape)
        print('Min shape', min_shape)

        print('Minimum feature length in bottleneck: ', self.unet_feature_map_min_edge_length)

        self.plans_per_stage = list()

        target_spacing_transposed = np.array(target_spacing)[self.transpose_forward]
        median_shape_transposed = np.array(median_shape)[self.transpose_forward]
        print("Transposed median shape of the dataset: ", median_shape_transposed)

        print("Generating configuration for full resolution network...")
        self.plans_per_stage.append(self.get_properties_for_stage(target_spacing_transposed, target_spacing_transposed,
                                                                  median_shape_transposed,
                                                                  len(sizes),
                                                                  num_modalities))

        architecture_input_voxels_here = np.prod(self.plans_per_stage[-1]['patch_size'], dtype=np.int64)

        if architecture_input_voxels_here / np.prod(median_shape) <= self.proportion_of_patient_visible_at_stage_0:
            print("Generating configuration for lowres")

            lowres_stage_spacing = deepcopy(target_spacing)
            num_voxels = np.prod(median_shape, dtype=np.float64)
            while architecture_input_voxels_here / num_voxels <= self.proportion_of_patient_visible_at_stage_0:
                max_spacing = max(lowres_stage_spacing)
                if np.any((max_spacing / lowres_stage_spacing) > 2):
                    lowres_stage_spacing[(max_spacing / lowres_stage_spacing) > 2] \
                        *= 1.01
                else:
                    lowres_stage_spacing *= 1.01
                num_voxels = np.prod(target_spacing / lowres_stage_spacing * median_shape, dtype=np.float64)

                lowres_stage_spacing_transposed = np.array(lowres_stage_spacing)[self.transpose_forward]
                new = self.get_properties_for_stage(lowres_stage_spacing_transposed, target_spacing_transposed,
                                                    median_shape_transposed,
                                                    len(sizes),
                                                    num_modalities)
                architecture_input_voxels_here = np.prod(new['patch_size'], dtype=np.int64)

            # Only add the low-res version if it covers more than double the voxels of the original
            if 2 * np.prod(new['median_patient_size_in_voxels'], dtype=np.int64) < np.prod(
                    self.plans_per_stage[0]['median_patient_size_in_voxels'], dtype=np.int64):
                self.plans_per_stage.append(new)

        self.plans_per_stage = self.plans_per_stage[::-1]
        self.plans_per_stage = {i: self.plans_per_stage[i] for i in range(len(self.plans_per_stage))}

        print(self.plans_per_stage)
        print("transpose forward", self.transpose_forward)
        print("transpose backward", self.transpose_backward)
        print("Number of disabled skip connections", self.num_disable_skip)

        channel_list = get_channel_list(modalities)
        normalization_schemes = OrderedDict()
        for i in range(len(channel_list)):
            normalization_schemes[i] = modality_norm_scheme(channel_list[i])

        plans = {'num_stages': len(list(self.plans_per_stage.keys())),
                 'num_modalities': num_modalities,
                 'modalities': modalities,
                 'normalization_schemes': normalization_schemes,
                 'dataset_properties': self.dataset_properties,
                 'original_spacings': spacings,
                 'original_sizes': sizes,
                 'preprocessed_data_folder': self.preprocessed_data_path,
                 'base_num_features': self.unet_base_num_features,
                 'transpose_forward': self.transpose_forward,
                 'transpose_backward': self.transpose_backward,
                 'data_identifier': self.data_identifier,
                 'plans_per_stage': self.plans_per_stage,
                 'conv_per_stage': self.conv_per_stage,
                 'num_disable_skip': self.num_disable_skip
                 }

        self.plans = plans
        self.save_my_plans()

    def run_preprocessing(self):

        modalities = self.plans['modalities']
        normalization_schemes = self.plans['normalization_schemes']
        sample_identifiers = self.plans['dataset_properties']['sample_identifiers']
        sample_properties = self.plans['dataset_properties']['sample_properties']
        intensity_properties = self.plans['dataset_properties']['intensity_properties']
        make_foreground_masks = self.plans['dataset_properties']['has_uniform_background']

        preprocessor = GenericPreprocessor(modalities, normalization_schemes, self.transpose_forward,
                                           intensity_properties, make_foreground_masks)
        target_spacings = [i["current_spacing"] for i in self.plans_per_stage.values()]

        if self.plans['num_stages'] > 1 and not isinstance(self.num_processes, (list, tuple)):
            curr_num_processes = (default_num_processes, self.num_processes)
        elif self.plans['num_stages'] == 1 and isinstance(self.num_processes, (list, tuple)):
            curr_num_processes = self.num_processes[-1]
        else:
            curr_num_processes = self.num_processes

        preprocessor.run(target_spacings, self.raw_data_path / 'imagesTr', sample_identifiers, sample_properties,
                         self.preprocessed_data_path, self.plans['data_identifier'], curr_num_processes)
