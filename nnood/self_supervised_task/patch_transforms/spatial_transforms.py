import itertools
from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import rotate, zoom

from nnood.self_supervised_task.patch_transforms.base_transform import PatchTransform
from nnood.self_supervised_task.utils import check_object_overlap, get_patch_slices, get_patch_image_slices


class ResizePatch(PatchTransform):

    def __init__(self, min_overlap_pct: Optional[float] = None, min_scale: float = 0.7, max_scale: float = 1.3,
                 validate_position=True):
        """
        :param min_overlap_pct: minimum percentage overlap between objects in patch and destination
        :param min_scale: minimum scaling to be applied
        :param max_scale: maximum scaling to be applied
        :param validate_position: You may want to disable position validation if you know you are going to randomly
                shift the patch afterwards
        """
        super(ResizePatch, self).__init__(min_overlap_pct)
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.validate_position = validate_position

    def transform(self, patch: np.ndarray, patch_mask: np.ndarray, patch_corner: np.ndarray, dest: np.ndarray,
                  dest_bbox_lbs: np.ndarray, dest_bbox_ubs: np.ndarray, patch_object_mask: Optional[np.ndarray],
                  dest_object_mask: Optional[np.ndarray]) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:

        if self.validate_position and patch_object_mask is not None:
            assert dest_object_mask is not None, 'Patch object mask is valid, but dest_object_mask is None in ' \
                                                 'ResizeTransform.'
            assert check_object_overlap(patch_corner, patch_mask, patch_object_mask, dest_object_mask,
                                        self.min_overlap_pct), 'Input for ResizeTransform doesn\'t satisfy object ' \
                                                               'overlap conditions.'

        patch_shape = np.array(patch_mask.shape)

        ub = self._calc_max_scale(patch_corner, patch_shape, dest_bbox_ubs) \
            if self.validate_position else np.minimum(self.max_scale,
                                                      np.min((dest_bbox_ubs - dest_bbox_lbs) / patch_shape)
                                                      - 0.02)  # Minus small epsilon to ensure we stay within bounds

        scale = np.clip(np.random.normal(1, 0.5), self.min_scale, ub)

        old_patch_half_width = patch_shape // 2

        new_patch = np.stack([zoom(patch[i], scale) for i in range(patch.shape[0])])
        new_patch_mask = zoom(patch_mask.astype(float), scale) > 0.5
        new_patch_corner = patch_corner + old_patch_half_width - np.array(new_patch_mask.shape) // 2
        new_patch_object_mask = patch_object_mask if patch_object_mask is None else \
            zoom(patch_object_mask.astype(float), scale) > 0.5

        if self.validate_position and patch_object_mask is not None and \
                not check_object_overlap(new_patch_corner, new_patch_mask, new_patch_object_mask, dest_object_mask,
                                         self.min_overlap_pct):
            # Should not need to track attempts, we know the condition holds before scaling, and we're using a scale
            # factor sampled around 1.
            return self.transform(patch, patch_mask, patch_corner, dest, dest_bbox_lbs, dest_bbox_ubs,
                                  patch_object_mask, dest_object_mask)

        assert np.all(np.array(new_patch_mask.shape) != 0), f'{new_patch_corner} {np.array(new_patch_mask)}'

        return new_patch, new_patch_mask, new_patch_corner, new_patch_object_mask

    def _calc_max_scale(self, patch_corner: np.ndarray, patch_shape: np.ndarray, max_coords: np.ndarray) -> float:
        patch_half_width = patch_shape // 2
        patch_centre = patch_corner + patch_half_width
        patch_top_corner = patch_corner + patch_shape

        centre_to_top = patch_top_corner - patch_centre
        centre_to_max = max_coords - patch_centre
        min_up_scale = min(centre_to_max / centre_to_top)
        min_down_scale = min(patch_centre / patch_half_width)
        return min(self.max_scale, min_up_scale, min_down_scale)


class TranslatePatch(PatchTransform):

    def transform(self, patch: np.ndarray, patch_mask: np.ndarray, patch_corner: np.ndarray, dest: np.ndarray,
                  dest_bbox_lbs: np.ndarray, dest_bbox_ubs: np.ndarray, patch_object_mask: Optional[np.ndarray],
                  dest_object_mask: Optional[np.ndarray]) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:

        max_coords = dest_bbox_ubs - np.array(patch_mask.shape)

        if (dest_bbox_lbs >= max_coords + 1).any():
            print('AAAAAAAAAAAAAAAAA')
        # Add one as randint uses exclusive upperbound
        new_patch_corner = np.random.randint(dest_bbox_lbs, max_coords + 1)

        if patch_object_mask is not None and dest_object_mask is not None and \
                not check_object_overlap(new_patch_corner, patch_mask, patch_object_mask, dest_object_mask,
                                         self.min_overlap_pct):
            # Try again if new location doesn't overlap objects enough
            return self.transform(patch, patch_mask, patch_corner, dest, dest_bbox_lbs, dest_bbox_ubs,
                                  patch_object_mask, dest_object_mask)
        else:
            return patch, patch_mask, new_patch_corner, patch_object_mask


class RotatePatch(PatchTransform):

    def __init__(self, min_overlap_pct: Optional[float] = None, validate_position=True):
        """
        :param min_overlap_pct: minimum percentage overlap between objects in patch and destination
        :param validate_position: You may want to disable position validation if you know you are going to randomly
                shift the patch afterwards
        """
        super(RotatePatch, self).__init__(min_overlap_pct)
        self.validate_position = validate_position
        self.rng = np.random.default_rng()

    def transform(self, patch: np.ndarray, patch_mask: np.ndarray, patch_corner: np.ndarray, dest: np.ndarray,
                  dest_bbox_lbs: np.ndarray, dest_bbox_ubs: np.ndarray, patch_object_mask: Optional[np.ndarray],
                  dest_object_mask: Optional[np.ndarray]) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:

        if self.validate_position and patch_object_mask is not None:
            assert dest_object_mask is not None, 'Patch object mask is valid, but dest_object_mask is None in ' \
                                                 'RotateTransform.'
            assert check_object_overlap(patch_corner, patch_mask, patch_object_mask, dest_object_mask,
                                        self.min_overlap_pct), 'Input for RotateTransform doesn\'t satisfy object ' \
                                                               'overlap conditions.'

        num_dims = len(patch_corner)
        patch_shape = np.array(patch_mask.shape)
        patch_centre = patch_corner + patch_shape // 2

        assert np.all(patch_shape != 0), f'{patch_corner} {patch_shape}'

        # Rotate patch, using all possible access combinations
        new_patch, new_patch_mask, new_patch_object_mask = patch, patch_mask.astype(float), patch_object_mask
        for ax1, ax2 in itertools.combinations(range(num_dims), 2):
            angle = self.rng.normal(scale=60)

            # Add one to axes, to account for channels dimension
            new_patch = rotate(new_patch, angle, axes=(ax1 + 1, ax2 + 1))
            ax_pair = (ax1, ax2)
            new_patch_mask = rotate(new_patch_mask, angle, axes=ax_pair)
            new_patch_object_mask = new_patch_object_mask if new_patch_object_mask is None else \
                rotate(new_patch_object_mask.astype(float), angle, axes=ax_pair)

        # Calculate new coord
        new_patch_shape = np.array(new_patch_mask.shape)
        new_patch_corner = patch_centre - new_patch_shape // 2
        new_patch_ub = new_patch_corner + new_patch_shape

        # Crop patch to be within bounds
        if self.validate_position and \
                (np.any(new_patch_corner < dest_bbox_lbs) or np.any(new_patch_ub > dest_bbox_ubs)):
            to_crop_below = np.maximum(dest_bbox_lbs - new_patch_corner, 0, dtype=int)
            to_crop_above = np.maximum(new_patch_ub - dest_bbox_ubs, 0, dtype=int)

            cropped_shape = new_patch_shape - to_crop_below - to_crop_above

            # Slightly abusing these functions, but it should work, as we are extracting a patch from within the patch
            new_patch = new_patch[get_patch_image_slices(to_crop_below, cropped_shape, new_patch.shape[0])]
            mask_slices = get_patch_slices(to_crop_below, cropped_shape)
            new_patch_mask = new_patch_mask[mask_slices]
            new_patch_corner += to_crop_below

            if new_patch_object_mask is not None:
                new_patch_object_mask = new_patch_object_mask[mask_slices]

        elif np.any(new_patch_shape > (dest_bbox_ubs - dest_bbox_lbs)):
            # Even if we're not validating the position, it must still be possible to fit the patch within the desired
            # bounds

            size_to_reduce = np.maximum(new_patch_shape - (dest_bbox_ubs - dest_bbox_lbs), 0)
            to_crop_below = size_to_reduce // 2

            # Again, abuse to reduce to valid size
            target_shape = new_patch_shape - size_to_reduce
            new_patch = new_patch[get_patch_image_slices(to_crop_below, target_shape, new_patch.shape[0])]
            mask_slices = get_patch_slices(to_crop_below, target_shape)
            new_patch_mask = new_patch_mask[mask_slices]
            # Don't need to update patch_corner, as not validating actual position

            if new_patch_object_mask is not None:
                new_patch_object_mask = new_patch_object_mask[mask_slices]

        new_patch_mask = new_patch_mask > 0.5
        if new_patch_object_mask is not None:
            new_patch_object_mask = new_patch_object_mask > 0.5

        if self.validate_position and patch_object_mask is not None and \
                not check_object_overlap(new_patch_corner, new_patch_mask, new_patch_object_mask, dest_object_mask,
                                         self.min_overlap_pct):
            # Should not need to track attempts, we know the condition holds before rotating, and we're using an angle
            # sampled around 1.
            return self.transform(patch, patch_mask, patch_corner, dest, dest_bbox_lbs, dest_bbox_ubs,
                                  patch_object_mask, dest_object_mask)

        assert np.all(np.array(new_patch_mask.shape) != 0), f'{new_patch_corner} {np.array(new_patch_mask.shape)}'

        return new_patch, new_patch_mask, new_patch_corner, new_patch_object_mask
