import itertools

import numpy as np
from skimage import filters, measure, morphology, segmentation

from nnood.utils.miscellaneous import make_hypersphere_mask


def get_object_mask(img):
    # Excluding channels dimension
    num_channels = img.shape[0]
    image_shape = np.array(img.shape)[1:]

    # 5% of each dimension length
    corner_lens = image_shape // 20

    # Used for upper_bounds, so use dim lengths
    img_corners = list(itertools.product(*[[0, s] for s in image_shape]))

    all_corner_ranges = [[(0, l) if c == 0 else (c - l, c) for c, l in zip(c_coord, corner_lens)]
                         for c_coord in img_corners]
    corner_patch_slices = [tuple([slice(lb, ub) for lb, ub in cr]) for cr in all_corner_ranges]

    num_corner_seed_points = 2 ** len(image_shape)

    masks = []

    for i in range(num_channels):

        sobel_channel = filters.sobel(img[i])

        curr_channel_masks = []

        for c_c, c_r, c_s in zip(img_corners, all_corner_ranges, corner_patch_slices):

            patch_tolerance = sobel_channel[c_s].std()

            for _ in range(num_corner_seed_points):
                random_c = tuple([np.random.randint(lb, ub) for lb, ub in c_r])

                curr_channel_masks.append(segmentation.flood(sobel_channel, random_c, tolerance=patch_tolerance))

        masks.append(np.any(np.stack(curr_channel_masks), axis=0))

    bg_mask = np.all(np.stack(masks), axis=0)
    fg_mask = np.logical_not(bg_mask)

    def get_biggest_connected_component(m):
        label_m = measure.label(m)
        region_sizes = np.bincount(label_m.flatten())
        # Zero size of background, so is ignored when finding biggest region
        region_sizes[0] = 0
        biggest_region_label = np.argmax(region_sizes)
        return label_m == biggest_region_label

    init_biggest_region = get_biggest_connected_component(fg_mask)

    # Apply binary opening to mask of largest object, to smooth out edges / disconnect any spurious
    opening_structure_r = max(np.median(image_shape) // 250, 1)
    opening_structure = make_hypersphere_mask(opening_structure_r, len(image_shape))
    opened_biggest_region = morphology.binary_opening(init_biggest_region, footprint=opening_structure)

    return get_biggest_connected_component(opened_biggest_region)
