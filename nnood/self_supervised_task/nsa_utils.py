import itertools
from multiprocessing import Pool

import numpy as np

from nnood.configuration import default_num_processes
from nnood.training.dataloading.dataset_loading import load_npy_or_npz


def get_avg_mask_bounds(m):
    non_zero_coords = np.where(m)
    dims = len(m.shape)

    unique_non_zero_coords = np.array([np.unique(cs) for cs in non_zero_coords],
                                      dtype=object)

    obj_avg_dim_lens = []

    for d in range(dims):
        other_ds = [d2 for d2 in range(dims) if d2 != d]

        other_coord_combs = itertools.product(*(unique_non_zero_coords[other_ds]))

        curr_obj_dim_lens = []

        # For each combination of other coordinates
        for coord_comb in other_coord_combs:

            test = [non_zero_coords[d2] == c2 for d2, c2 in zip(other_ds, coord_comb)]
            # Get indices of coordinates which match this combination
            coord_inds = np.all(test,
                                axis=0)
            if not coord_inds.any():
                continue

            # Get corresponding coordinates. Min and Max are beginning and end,
            # due to ordering made by flattening
            min_c, max_c = non_zero_coords[d][coord_inds][[0, -1]]

            curr_obj_dim_lens.append(max_c - min_c + 1)

        obj_avg_dim_lens.append(np.mean(curr_obj_dim_lens) / m.shape[d])

    return obj_avg_dim_lens


def load_mask_and_get_stats(f):
    _, s_mask = load_npy_or_npz(f, 'r', True)

    return get_avg_mask_bounds(s_mask), np.mean(s_mask)


def nsa_sample_dimension(lb, ub, img_d):
    gamma_lb = 0.03
    gamma_shape = 2
    gamma_scale = 0.1

    gamma_sample = (gamma_lb + np.random.gamma(gamma_shape, gamma_scale)) * img_d

    return int(np.clip(gamma_sample, lb, ub))


def compute_nsa_mask_params(class_has_foreground, dataset, data_num_dims):
    if class_has_foreground:

        with Pool(default_num_processes) as pool:
            mask_stats = pool.map(load_mask_and_get_stats, [v['data_file'] for v in dataset.values()])

        # Average proportional length of object along each dimension
        avg_obj_dim_len = np.mean([m_s[0] for m_s in mask_stats], axis=0)
        # Average area of object proportional to entire image
        avg_obj_area = np.mean([m_s[1] for m_s in mask_stats])

        width_bounds_pct = [(0.06, np.clip(d_len * 4 / 3, 0.25, 0.8)) for d_len in avg_obj_dim_len]
        num_patches = 3 if avg_obj_area < 0.75 else 4
        min_obj_pct = 0.5 if avg_obj_area < 0.4 else 0.7

    else:
        width_bounds_pct = [(0.06, 0.8)] * data_num_dims
        num_patches = 4
        min_obj_pct = None

    return width_bounds_pct, num_patches, min_obj_pct