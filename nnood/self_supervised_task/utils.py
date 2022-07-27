from typing import Tuple

import numpy as np


def check_object_overlap(patch_corner: np.ndarray, patch_mask: np.ndarray, patch_object_mask: np.ndarray,
                         dest_object_mask: np.ndarray, min_overlap_pct: float):
    # If no overlap required, skip calculations and return true.
    if min_overlap_pct == 0:
        return True

    dest_object_mask = extract_dest_patch_mask(patch_corner, patch_mask.shape, dest_object_mask)
    patch_and_object = patch_mask & patch_object_mask
    patch_and_dest = patch_and_object & dest_object_mask

    patch_obj_dst_area = np.sum(patch_and_dest)
    patch_obj_area = np.sum(patch_and_object)

    # Avoid division by zero. If the patch covers none of source object, we want to reject.
    if patch_obj_area == 0:
        return False

    return (patch_obj_dst_area / patch_obj_area) >= min_overlap_pct


def extract_dest_patch_mask(patch_corner: np.ndarray, patch_shape: Tuple[int], dest_object_mask: np.ndarray):
    assert len(patch_corner) == len(dest_object_mask.shape), 'Patch coordinate and destination object mask must have' \
                                                             f'equal number of dimensions: {patch_corner}, ' \
                                                             f'{dest_object_mask.shape}'
    assert len(patch_corner) == len(dest_object_mask.shape), 'Patch coordinate and patch shape must have equal ' \
                                                             f'number of dimensions: {patch_corner}, {patch_shape}'

    return dest_object_mask[get_patch_slices(patch_corner, patch_shape)]


def get_patch_slices(patch_corner: np.ndarray, patch_shape: Tuple[int]) -> Tuple[slice]:
    return tuple([slice(c, c + d) for (c, d) in zip(patch_corner, patch_shape)])


# Same as above, but with additional slice at beginning to include all image channels.
def get_patch_image_slices(patch_corner: np.ndarray, patch_shape: Tuple[int], img_channels: int) -> Tuple[slice]:
    return tuple([slice(img_channels)] + list(get_patch_slices(patch_corner, patch_shape)))
