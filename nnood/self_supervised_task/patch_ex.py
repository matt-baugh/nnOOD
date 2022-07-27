from typing import List, Optional, Tuple, Union

import numpy as np

from nnood.self_supervised_task.patch_blender import PatchBlender, SwapPatchBlender
from nnood.self_supervised_task.patch_labeller import PatchLabeller, BinaryPatchLabeller
from nnood.self_supervised_task.patch_shape_maker import PatchShapeMaker, EqualUniformPatchMaker
from nnood.self_supervised_task.patch_transforms.base_transform import PatchTransform
from nnood.self_supervised_task.utils import check_object_overlap, get_patch_slices, get_patch_image_slices


def patch_ex(img_dest: np.ndarray, img_src: Optional[np.ndarray] = None, same: bool = False, num_patches: int = 1,
             shape_maker: PatchShapeMaker = EqualUniformPatchMaker(), patch_transforms: List[PatchTransform] = [],
             blender: PatchBlender = SwapPatchBlender(), labeller: PatchLabeller = BinaryPatchLabeller(),
             return_anomaly_locations: bool = False, binary_factor: bool = True, min_overlap_pct: float = 0.25,
             width_bounds_pct: Union[Tuple[float, float], List[Tuple[float, float]]] = (0.05, 0.4),
             min_object_pct: float = 0.25, dest_bbox: Optional[np.ndarray] = None, extract_within_bbox: bool = False,
             skip_background: Optional[Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]] = None, verbose=True) \
        -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, List[np.ndarray]]]:
    """
    Create a synthetic training example from the given images by pasting/blending random patches.
    Args:
        :param img_dest: Image with shape (C[,Z],H,W) where patch should be changed.
        :param img_src: Optional, equal dimensions to img_dest, otherwise use ima_dest as source.
        :param same: Use ima_dest as source even if ima_src given.
        :param num_patches: How many patches to add. the method will always attempt to add the first patch,
                    for each subsequent patch it flips a coin.
        :param shape_maker: Used to create initial patch mask.
        :param patch_transforms: List of transforms to be applied to patch after it is selected with shape_maker.
        :param blender: Used to blend patches into destination image.
        :param labeller: Used to produce the label for the blended image.
        :param return_anomaly_locations: Bool for whether we return the list of anomaly centre coordinates.
        :param binary_factor: Whether to use a binary factor or sample from np.random.uniform(0.05, 0.95)
        :param min_overlap_pct: Minimum percentage object in patch extracted from source image and object in destination
                    image should overlap.
        :param width_bounds_pct: Either tuple of limits for every dimension, or list of tuples of limits for each
                    dimension.
        :param min_object_pct: Minimum percentage of object which the patch extracted from the source image must cover.
        :param dest_bbox: Shape (num_dims, 2). Specify which region to apply the anomalies within. If None, treat as
                    entire image.
        :param extract_within_bbox: Specify whether patches are extracted from inside dest_bbox from the source image.
        :param skip_background : Optional tuple of foreground masks for dest and src, or if 'same' then only src object
                    mask.
        :param verbose: Allows debugging prints.
    """
    img_src = img_dest.copy() if same or (img_src is None) else img_src

    if skip_background is not None:
        if same:
            dest_object_mask = src_object_mask = skip_background
        else:
            dest_object_mask = skip_background[0]
            src_object_mask = skip_background[1]

    else:
        dest_object_mask = None
        src_object_mask = None

    # add patches
    mask = np.zeros_like(img_dest[0], dtype=bool)  # single channel
    blended_img = img_dest.copy()

    if isinstance(width_bounds_pct, tuple):
        width_bounds_pct = [width_bounds_pct] * len(mask.shape)

    # Shape (spatial_dimensions, 2)
    width_bounds_pct = np.array(width_bounds_pct)

    if binary_factor:
        factor = 1.0
    else:
        factor = np.random.uniform(0.05, 0.95)

    anomaly_centres = []

    dest_bbox_lbs = np.zeros(len(img_dest.shape) - 1, dtype=int) if dest_bbox is None else dest_bbox[:, 0]
    dest_bbox_ubs = np.array(img_dest.shape[1:]) if dest_bbox is None else dest_bbox[:, 1]

    for i in range(num_patches):
        if i == 0 or np.random.randint(2) > 0:  # at least one patch
            blended_img, patch_corner, patch_mask = _patch_ex(
                blended_img, img_src, dest_object_mask, src_object_mask, shape_maker, patch_transforms, blender,
                width_bounds_pct, min_object_pct, min_overlap_pct, factor, verbose, dest_bbox_lbs, dest_bbox_ubs,
                extract_within_bbox)

            if patch_mask is not None:
                assert patch_corner is not None, 'patch_mask is not None, but patch_corner is???'
                assert patch_mask is not None, 'Should never be triggered, just for nice typing :)'

                mask[get_patch_slices(patch_corner, patch_mask.shape)] |= patch_mask

                anomaly_centres.append(patch_corner + np.array(patch_mask.shape) // 2)

    mask = mask.astype(float)
    final_label = labeller(factor, blended_img, img_dest, mask)[None]

    if return_anomaly_locations:
        # Convert label to single channel, to match network output
        return blended_img, final_label, anomaly_centres
    else:
        return blended_img, final_label


def _patch_ex(ima_dest: np.ndarray, ima_src: np.ndarray, dest_object_mask: Optional[np.ndarray],
              src_object_mask: Optional[np.ndarray], shape_maker: PatchShapeMaker,
              patch_transforms: List[PatchTransform], blender: PatchBlender,
              width_bounds_pct: np.ndarray, min_object_pct: float, min_overlap_pct: float, factor: float,
              verbose: bool, dest_bbox_lbs: np.ndarray, dest_bbox_ubs: np.ndarray, extract_within_bbox: bool) \
        -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    skip_background = (src_object_mask is not None) and (dest_object_mask is not None)
    dims = np.array(ima_dest.shape)
    bbox_shape = dest_bbox_ubs - dest_bbox_lbs

    min_dim_lens = (width_bounds_pct[:, 0] * bbox_shape).round().astype(int)
    max_dim_lens = (width_bounds_pct[:, 1] * bbox_shape).round().astype(int)
    dim_bounds = list(zip(min_dim_lens, max_dim_lens))

    patch_mask = shape_maker(dim_bounds, bbox_shape)

    found_patch = False
    attempts = 0

    src_patch_lb = dest_bbox_lbs if extract_within_bbox else np.zeros(len(ima_src.shape) - 1)
    src_patch_ub = dest_bbox_ubs if extract_within_bbox else np.array(ima_src.shape[1:])

    # Use minimum patch size as buffer to stop patch being too close to edge
    patch_centre_bounds = [(lb + b, ub - b) for (b, lb, ub) in zip(min_dim_lens, src_patch_lb, src_patch_ub)]

    if skip_background:
        # Reduce search space, so patch centre is within object bounding box. Reduces change of missing object, and
        # requiring more iterations.

        for d in range(len(patch_centre_bounds)):
            curr_lb, curr_ub = patch_centre_bounds[d]
            other_dims = tuple([d2 for d2 in range(len(patch_centre_bounds)) if d2 != d])
            obj_m_min_ind, obj_m_max_ind = np.nonzero(np.any(src_object_mask, axis=other_dims))[0][[0, -1]]

            patch_centre_bounds[d] = (max(curr_lb, obj_m_min_ind), min(curr_ub, obj_m_max_ind))

    while not found_patch:

        centers = np.array([np.random.randint(lb, ub) for lb, ub in patch_centre_bounds])
        patch_dims = np.array(patch_mask.shape)

        # Indices of patch corners relative to source, could be out of bounds!
        min_corner = centers - patch_dims // 2
        max_corner = min_corner + patch_dims

        # Indices of valid area WITHIN patch
        patch_min_indices = np.maximum(-min_corner, 0)
        patch_max_indices = patch_dims - np.maximum(max_corner - dims[1:], 0)

        test_patch_mask = patch_mask[tuple([slice(lb, ub) for (lb, ub) in zip(patch_min_indices, patch_max_indices)])]
        test_patch_corner = np.maximum(min_corner, 0)

        if skip_background:
            test_patch_object_mask = src_object_mask[get_patch_slices(test_patch_corner, test_patch_mask.shape)]
            object_area = np.sum(test_patch_mask & test_patch_object_mask)
            obj_area_sat = (object_area / np.prod(test_patch_mask.shape) > min_object_pct)

            # Want both conditions to hold. If first fails, skip second and iterate faster
            if obj_area_sat:
                found_patch = check_object_overlap(test_patch_corner, test_patch_mask, test_patch_object_mask,
                                                   dest_object_mask, min_overlap_pct)
            else:
                found_patch = False
        else:
            found_patch = True
        attempts += 1
        if attempts == 200:
            if verbose:
                print('No suitable patch found (initial location failed).')
            return ima_dest.copy(), None, None

    patch = ima_src[get_patch_image_slices(test_patch_corner, test_patch_mask.shape, ima_src.shape[0])]
    patch_mask = test_patch_mask
    patch_corner = test_patch_corner
    patch_object_mask = src_object_mask if src_object_mask is None or not skip_background else test_patch_object_mask

    for p_t in patch_transforms:
        patch, patch_mask, patch_corner, patch_object_mask = \
            p_t(patch, patch_mask, patch_corner, ima_dest, dest_bbox_lbs, dest_bbox_ubs, patch_object_mask,
                dest_object_mask)

    blended_img, patch_mask = blender(factor, patch, patch_mask, patch_corner, ima_dest, patch_object_mask,
                                      dest_object_mask)

    return blended_img, patch_corner, patch_mask


if __name__ == '__main__':
    patch_ex(np.random.random((3, 50, 50)), np.random.random((3, 50, 50)))
