import os
from copy import deepcopy

import numpy as np
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.color_transforms import GammaTransform
from batchgenerators.transforms.utility_transforms import RenameTransform, NumpyToTensor, AppendChannelsTransform

from nnood.training.data_augmentation.custom_transforms import Convert3DTo2DTransform, Convert2DTo3DTransform,\
    MySpatialTransform, MyMirrorTransform
from nnood.training.data_augmentation.downsampling import DownsampleSegForDSTransform
from nnood.training.dataloading.dataset_loading import DataLoader

default_3D_augmentation_params = {
    'do_elastic': False,
    'elastic_deform_alpha': (0., 900.),
    'elastic_deform_sigma': (9., 13.),
    'p_eldef': 0.2,

    'do_scaling': False,
    'scale_range': (1., 1.),
    'independent_scale_factor_for_each_axis': False,
    'p_independent_scale_per_axis': 1,
    'p_scale': 0.2,

    'do_rotation': False,
    'rotation_x': (0., 0.),
    'rotation_y': (0., 0.),
    'rotation_z': (0., 0.),
    'rotation_p_per_axis': 1,
    'p_rot': 0.2,

    'do_gamma': False,
    'gamma_retain_stats': True,
    'gamma_range': (0.7, 1.5),
    'p_gamma': 0.3,

    'do_mirror': False,
    'mirror_axes': (0, 1, 2),

    'dummy_2D': False,
    'mask_was_used_for_normalization': None,
    'border_mode_data': 'mirror',

    'all_segmentation_labels': None,  # used for cascade
    'move_last_seg_channel_to_data': False,  # used for cascade
    'cascade_do_cascade_augmentations': False,  # used for cascade
    'cascade_random_binary_transform_p': 0.4,
    'cascade_random_binary_transform_p_per_label': 1,
    'cascade_random_binary_transform_size': (1, 8),
    'cascade_remove_conn_comp_p': 0.2,
    'cascade_remove_conn_comp_max_size_percent_threshold': 0.15,
    'cascade_remove_conn_comp_fill_with_other_class_p': 0.0,

    'do_additive_brightness': False,
    'additive_brightness_p_per_sample': 0.15,
    'additive_brightness_p_per_channel': 0.5,
    'additive_brightness_mu': 0.0,
    'additive_brightness_sigma': 0.1,

    'num_threads': 12 if 'nnood_n_proc_DA' not in os.environ else int(os.environ['nnood_n_proc_DA']),
    'num_cached_per_thread': 1,
}

default_2D_augmentation_params = deepcopy(default_3D_augmentation_params)

default_2D_augmentation_params['elastic_deform_alpha'] = (0., 200.)
default_2D_augmentation_params['elastic_deform_sigma'] = (9., 13.)

# Sometimes you have 3d data and a 3d net but cannot augment them properly in 3d due to anisotropy (which is currently
# not supported in batchgenerators). In that case you can 'cheat' and transfer your 3d data into 2d data and
# transform them back after augmentation
default_2D_augmentation_params['dummy_2D'] = False
default_2D_augmentation_params['mirror_axes'] = (0, 1)  # this can be (0, 1, 2) if dummy_2D=True


def get_patch_size(final_patch_size, rot_x, rot_y, rot_z, scale_range):
    if isinstance(rot_x, (tuple, list)):
        rot_x = max(np.abs(rot_x))
    if isinstance(rot_y, (tuple, list)):
        rot_y = max(np.abs(rot_y))
    if isinstance(rot_z, (tuple, list)):
        rot_z = max(np.abs(rot_z))
    rot_x = min(90 / 360 * 2. * np.pi, rot_x)
    rot_y = min(90 / 360 * 2. * np.pi, rot_y)
    rot_z = min(90 / 360 * 2. * np.pi, rot_z)
    from batchgenerators.augmentations.utils import rotate_coords_3d, rotate_coords_2d
    coords = np.array(final_patch_size)
    final_shape = np.copy(coords)
    if len(coords) == 3:
        final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, rot_x, 0, 0)), final_shape)), 0)
        final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, 0, rot_y, 0)), final_shape)), 0)
        final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, 0, 0, rot_z)), final_shape)), 0)
    elif len(coords) == 2:
        final_shape = np.max(np.vstack((np.abs(rotate_coords_2d(coords, rot_x)), final_shape)), 0)
    final_shape /= min(scale_range)
    return final_shape.astype(int)


def get_default_augmentation(dataloader_train: DataLoader, dataloader_val: DataLoader, patch_size, label_is_seg,
                             params=default_3D_augmentation_params, border_val_seg=0, pin_memory=True,
                             seeds_train=None, seeds_val=None, deep_supervision_scales=None):

    tr_transforms = []

    # don't do color augmentations while in 2d mode with 3d data because the color channel is overloaded!!
    if params.get('dummy_2D') is not None and params.get('dummy_2D'):
        # 3DTo2D also flatten positional encoding
        tr_transforms.append(Convert3DTo2DTransform())
        patch_size_spatial = patch_size[1:]
    else:
        patch_size_spatial = patch_size

    # Reimplementation of SpatialTransformer from batchgenerators, now treating 'seg' as float valued score.
    # Also disable random cropping (as in patch based framework), and update positional encoding to match spatial
    # transformations (see comments in implementation for more details).
    tr_transforms.append(MySpatialTransform(
        patch_size_spatial, label_is_seg=label_is_seg, patch_center_dist_from_border=None,
        do_elastic_deform=params.get('do_elastic'), alpha=params.get('elastic_deform_alpha'),
        sigma=params.get('elastic_deform_sigma'), do_rotation=params.get('do_rotation'),
        angle_x=params.get('rotation_x'), angle_y=params.get('rotation_y'), angle_z=params.get('rotation_z'),
        do_scale=params.get('do_scaling'), scale=params.get('scale_range'),
        border_mode_data=params.get('border_mode_data'), border_cval_data=0, order_data=3, border_mode_seg='mirror',
        border_cval_seg=border_val_seg,
        order_seg=1, random_crop=False, p_el_per_sample=params.get('p_eldef'),
        p_scale_per_sample=params.get('p_scale'), p_rot_per_sample=params.get('p_rot'),
        independent_scale_for_each_axis=params.get('independent_scale_factor_for_each_axis')
    ))

    if params.get('dummy_2D') is not None and params.get('dummy_2D'):
        # 2DTo3D also restores positional encoding
        tr_transforms.append(Convert2DTo3DTransform())

    if params.get('do_mirror'):
        # This does update the positional encoding according to any mirroring which takes place, as we assume allowing
        # mirroring in a certain axes means that the current image could 'normally' occur flipped along that axis.
        tr_transforms.append(MyMirrorTransform(params.get('mirror_axes')))

    if params.get('do_gamma'):
        tr_transforms.append(
            GammaTransform(params.get('gamma_range'), False, True, retain_stats=params.get('gamma_retain_stats'),
                           p_per_sample=params['p_gamma']))

    # As we don't have a segmentation to indicate where the background is, we can't do this atm
    # commenting as we may want to implement something heuristic later down the line.
    # if params.get('mask_was_used_for_normalization') is not None:
    #     mask_was_used_for_normalization = params.get('mask_was_used_for_normalization')
    #     tr_transforms.append(MaskTransform(mask_was_used_for_normalization, mask_idx_in_seg=0, set_outside_to=0))

    # Now all the data transformations are complete, append the entire positional encoding to the data.
    tr_transforms.append(AppendChannelsTransform('coords', 'data', list(range(len(patch_size)))))

    if params.get('move_last_seg_channel_to_data') is not None and params.get('move_last_seg_channel_to_data'):
        # Move score from previous level back to data
        if label_is_seg:
            # TODO: implement this, low priority as all self-sup tasks are regression based atm
            assert False, 'Cascade of segmentations not yet implemented, need to know number of labels'
            # tr_transforms.append(MoveSegAsOneHotToData(1, params.get("all_segmentation_labels"), 'seg', 'data'))
        else:
            tr_transforms.append(AppendChannelsTransform('seg', 'data', [1]))

        if params.get('cascade_do_cascade_augmentations') is not None and params.get(
                'cascade_do_cascade_augmentations'):
            assert False, 'You\'ve tried to run cascade-specific augmentations, but I haven\' implemented it yet!'
            # nnUNet removed random connected component, or applied random binary transform
            # not sure what is appropriate for our task, but will require custom transform

    tr_transforms.append(RenameTransform('seg', 'target', True))

    if deep_supervision_scales is not None:
        tr_transforms.append(DownsampleSegForDSTransform(deep_supervision_scales, 1, input_key='target',
                                                         output_key='target'))

    tr_transforms.append(NumpyToTensor(['data', 'target'], 'float'))

    tr_transforms = Compose(tr_transforms)

    batchgenerator_train = MultiThreadedAugmenter(dataloader_train, tr_transforms, params.get('num_threads'),
                                                  params.get('num_cached_per_thread'), seeds=seeds_train,
                                                  pin_memory=pin_memory)

    # Now all the data transformations are complete, append the entire positional encoding to the data.
    val_transforms = [AppendChannelsTransform('coords', 'data', list(range(len(patch_size))))]

    if params.get('move_last_seg_channel_to_data') is not None and params.get('move_last_seg_channel_to_data'):
        # Move score from previous level back to data
        if label_is_seg:
            # TODO: implement this, low priority as all self-sup tasks are regression based atm
            assert False, 'Cascade for validation segmentations not yet implemented, need to know number of labels'
            # val_transforms.append(MoveSegAsOneHotToData(1, params.get("all_segmentation_labels"), 'seg', 'data'))

        else:
            val_transforms.append(AppendChannelsTransform('seg', 'data', [1]))

    val_transforms.append(RenameTransform('seg', 'target', True))

    if deep_supervision_scales is not None:
        val_transforms.append(DownsampleSegForDSTransform(deep_supervision_scales, 1, input_key='target',
                                                          output_key='target'))

    val_transforms.append(NumpyToTensor(['data', 'target'], 'float'))

    val_transforms = Compose(val_transforms)

    batchgenerator_val = MultiThreadedAugmenter(dataloader_val, val_transforms, max(params.get('num_threads') // 2, 1),
                                                params.get('num_cached_per_thread'), seeds=seeds_val,
                                                pin_memory=pin_memory)
    return batchgenerator_train, batchgenerator_val
