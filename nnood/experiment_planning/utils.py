from copy import deepcopy

import numpy as np


def get_pool_and_conv_props(patch_size: np.ndarray, min_feature_map_size: int, max_num_pool: int, spacing: np.ndarray):
    """
    :param patch_size:
    :param min_feature_map_size: min edge length of feature maps in bottleneck
    :param max_num_pool
    :param spacing:
    :return:
    """
    dim = len(spacing)

    current_spacing = deepcopy(list(spacing))
    current_size = deepcopy(list(patch_size))

    pool_op_kernel_sizes = []
    conv_kernel_sizes = []

    num_pool_per_axis = [0] * dim

    while True:
        min_spacing = min(current_spacing)
        valid_axes_for_pool = [i for i in range(dim) if current_spacing[i] / min_spacing < 2]
        axes = []
        for a in range(dim):
            my_spacing = current_spacing[a]
            partners = [i for i in range(dim) if
                        current_spacing[i] / my_spacing < 2 and my_spacing / current_spacing[i] < 2]
            if len(partners) > len(axes):
                axes = partners
        conv_kernel_size = [3 if i in axes else 1 for i in range(dim)]

        # Exclude axes which cannot be pooled further
        valid_axes_for_pool = [i for i in valid_axes_for_pool if current_size[i] >= 2 * min_feature_map_size
                               and num_pool_per_axis[i] < max_num_pool]

        if len(valid_axes_for_pool) == 0:
            break

        other_axes = [i for i in range(dim) if i not in valid_axes_for_pool]

        pool_kernel_sizes = [0] * dim
        for v in valid_axes_for_pool:
            pool_kernel_sizes[v] = 2
            num_pool_per_axis[v] += 1
            current_spacing[v] *= 2
            current_size[v] = np.ceil(current_size[v] / 2)
        for nv in other_axes:
            pool_kernel_sizes[nv] = 1

        pool_op_kernel_sizes.append(pool_kernel_sizes)
        conv_kernel_sizes.append(conv_kernel_size)

    must_be_divisible_by = get_shape_must_be_divisible_by(num_pool_per_axis)
    patch_size = pad_shape(patch_size, must_be_divisible_by)

    # Add bottleneck conv
    conv_kernel_sizes.append([3] * dim)

    # Note: this computes the number of conv/pools for one side of the U-Net, not both
    return num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, patch_size, must_be_divisible_by


def get_shape_must_be_divisible_by(net_numpool_per_axis):
    return 2 ** np.array(net_numpool_per_axis)


def pad_shape(shape, must_be_divisible_by):
    """
    pads shape so that it is divisibly by must_be_divisible_by
    :param shape:
    :param must_be_divisible_by:
    :return:
    """
    if not isinstance(must_be_divisible_by, (tuple, list, np.ndarray)):
        must_be_divisible_by = [must_be_divisible_by] * len(shape)
    else:
        assert len(must_be_divisible_by) == len(shape)

    new_shape = [shape[i] + must_be_divisible_by[i] - shape[i] % must_be_divisible_by[i] for i in range(len(shape))]

    for i in range(len(shape)):
        if shape[i] % must_be_divisible_by[i] == 0:
            new_shape[i] -= must_be_divisible_by[i]
    new_shape = np.array(new_shape).astype(int)
    return new_shape


def summarise_plans(plans):
    print("modalities: ", plans['modalities'])
    print("normalization_schemes", plans['normalization_schemes'])
    print("stages...\n")

    for i in range(len(plans['plans_per_stage'])):
        print("stage: ", i)
        print(plans['plans_per_stage'][i])
        print("")
