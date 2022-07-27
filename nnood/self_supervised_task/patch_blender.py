from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np
import cv2
from pietorch import blend_dst_numpy

from nnood.preprocessing.normalisation import _norm_helper
from nnood.self_supervised_task.utils import extract_dest_patch_mask, get_patch_image_slices


class PatchBlender(ABC):

    @abstractmethod
    def blend(self, factor: float, patch: np.ndarray, patch_mask: np.ndarray, patch_corner: np.ndarray,
              dest: np.ndarray, patch_object_mask: Optional[np.ndarray], dest_object_mask: Optional[np.ndarray]) \
            -> Tuple[np.ndarray, np.ndarray]:
        """
        :param factor: Extent that patch is used in blending. In range [0-1]. May be ignored.
        :param patch: Patch extracted from source image.
        :param patch_mask: Mask of which elements of patch will be put/blended into the destination image.
        :param patch_corner: Coordinate of the minimum element of patch relative to the destination image.
        :param dest: Destination image, available for querying but probably shouldn't change!
        :param patch_object_mask: Mask showing which elements of the patch contain an object of interest.
        :param dest_object_mask: Mask showing which elements of the destination image contain an object of interest.
        :returns Tuple of blended image and final patch mask.
        """
        pass

    def __call__(self, factor: float, patch: np.ndarray, patch_mask: np.ndarray, patch_corner: np.ndarray,
                 dest: np.ndarray, patch_object_mask: Optional[np.ndarray], dest_object_mask: Optional[np.ndarray]) \
            -> Tuple[np.ndarray, np.ndarray]:
        return self.blend(factor, patch, patch_mask, patch_corner, dest, patch_object_mask, dest_object_mask)

    @staticmethod
    def get_final_patch_mask(patch_mask: np.ndarray, patch_corner: np.ndarray, patch_object_mask: np.ndarray,
                             dest_object_mask: np.ndarray) -> np.ndarray:
        return patch_mask & (patch_object_mask | extract_dest_patch_mask(patch_corner, patch_mask.shape,
                                                                         dest_object_mask))


class SwapPatchBlender(PatchBlender):
    def blend(self, _: float, patch: np.ndarray, patch_mask: np.ndarray, patch_corner: np.ndarray,
              dest: np.ndarray, patch_object_mask: Optional[np.ndarray], dest_object_mask: Optional[np.ndarray]) \
            -> Tuple[np.ndarray, np.ndarray]:
        if patch_object_mask is not None and dest_object_mask is not None:
            patch_mask = self.get_final_patch_mask(patch_mask, patch_corner, patch_object_mask, dest_object_mask)

        image_slices = get_patch_image_slices(patch_corner, patch_mask.shape, patch.shape[0])
        blended_img = dest.copy()
        before = dest[image_slices]
        blended_img[image_slices] -= patch_mask * before
        blended_img[image_slices] += patch_mask * patch
        return blended_img, patch_mask


class UniformPatchBlender(PatchBlender):
    def blend(self, factor: float, patch: np.ndarray, patch_mask: np.ndarray, patch_corner: np.ndarray,
              dest: np.ndarray, patch_object_mask: Optional[np.ndarray], dest_object_mask: Optional[np.ndarray]) \
            -> Tuple[np.ndarray, np.ndarray]:
        if patch_object_mask is not None and dest_object_mask is not None:
            patch_mask = self.get_final_patch_mask(patch_mask, patch_corner, patch_object_mask, dest_object_mask)

        image_slices = get_patch_image_slices(patch_corner, patch_mask.shape, patch.shape[0])
        blended_img = dest.copy()
        before = dest[image_slices]
        blended_img[image_slices] -= factor * patch_mask * before
        blended_img[image_slices] += factor * patch_mask * patch
        return blended_img, patch_mask


# This should not be used for training, only implemented to later compare with own, generic poisson image editing
# implementation
class OpenCVPoissonPatchBlender(PatchBlender):

    def __init__(self, mode=cv2.NORMAL_CLONE, norm_args=None):
        self.mode = mode
        # Custom normalisation arguments, to be used with _norm_helper
        self.norm_args = norm_args

    def blend(self, factor: float, patch: np.ndarray, patch_mask: np.ndarray, patch_corner: np.ndarray,
              dest: np.ndarray, patch_object_mask: Optional[np.ndarray], dest_object_mask: Optional[np.ndarray]) \
            -> Tuple[np.ndarray, np.ndarray]:

        assert len(dest.shape) == 3 and dest.shape[0] in [1, 3], 'OpenCV patch blending only works on 3 or 1 ' \
                                                                 'channeled, 2D images.'
        assert len(patch.shape) == 3 and patch.shape[0] in [1, 3], 'OpenCV patch blending only works on 3 or 1 ' \
                                                                   'channeled, 2D images.'

        blended_img = dest.copy()

        norm_scheme = None
        init_dtype = blended_img.dtype

        if self.norm_args is not None:
            # Assume custom mean/std denormalises to 0-1. Still need to convert to 0-255
            patch = np.uint8(np.round(255 * _norm_helper(patch, *self.norm_args, False)))
            blended_img = np.uint8(np.round(255 * _norm_helper(blended_img, *self.norm_args, False)))

            if dest.shape[0] == 1:
                patch = np.repeat(patch, 3, axis=0)
                blended_img = np.repeat(blended_img, 3, axis=0)

        elif init_dtype is not np.uint8:
            # Need to convert patch and blended_img to uint8, as OpenCV only takes those dtypes.
            blended_img_min, blended_img_max = np.min(blended_img), np.max(blended_img)

            if 0 <= blended_img_min:
                if blended_img_max <= 1:
                    print('Assuming images are normalised to be in [0-1] range')
                    norm_scheme = '0-1'
                    patch = np.uint8(np.round(255 * patch))
                    blended_img = np.uint8(np.round(255 * blended_img))
                elif blended_img_max <= 255:
                    print('Assuming images are in range [0-255] and just need rounding')
                    norm_scheme = '0-255'
                    patch = np.uint8(np.round(patch))
                    blended_img = np.uint8(np.round(blended_img))
                else:
                    print('Failed to normalise image, returning unchanged image.')
                    return dest.copy(), np.zeros_like(patch_mask)
            else:
                print('Assuming images are normalised using ImageNet statistics')
                norm_scheme = 'imagenet'

                patch = np.uint8(np.round(255 * _norm_helper(patch, ['png-r', 'png-g', 'png-b'], None, None, False)))
                blended_img = np.uint8(np.round(255 * _norm_helper(blended_img, ['png-r', 'png-g', 'png-b'], None, None,
                                                                   False)))

        patch_mask_scaled = np.uint8(np.ceil(factor * 255) * patch_mask)

        # zero border to avoid artefacts
        patch_mask_scaled[0] = patch_mask_scaled[-1] = patch_mask_scaled[:, 0] = patch_mask_scaled[:, -1] = 0

        # cv2 seamlessClone will fail if positive mask area is too small
        if np.sum(patch_mask_scaled > 0) < 50:
            print('Masked area is too small to perform poisson image editing.')
            return dest.copy(), np.zeros_like(patch_mask)

        # Coordinates are reversed, because OpenCV expects images to be [H, W, C], yet for coordinates expects (x, y)
        # why are you like this OpenCV
        centre = tuple((patch_corner + np.array(patch_mask.shape) // 2)[::-1])

        # Move to channels last for opencv
        blended_img = np.moveaxis(blended_img, 0, -1)
        patch = np.moveaxis(patch, 0, -1)

        try:
            blended_img = cv2.seamlessClone(patch, blended_img, patch_mask_scaled, centre, self.mode)
        except cv2.error as e:
            print('WARNING, tried bad interpolation mask and got:', e)
            print('Info dump:')
            print('Dest orig shape: ', dest.shape)
            print('Dest curr shape: ', blended_img.shape)
            print('Patch shape (after moving axis): ', patch.shape)
            print('Patch mask shape: ', patch_mask.shape)
            print('Scaled patch mask shape: ', patch_mask_scaled.shape)
            print('Patch corner: ', patch_corner)
            print('OpenCV centre: ', centre)
            return dest.copy(), np.zeros_like(patch_mask)

        # Switch channels back
        blended_img = np.moveaxis(blended_img, -1, 0)

        if self.norm_args is not None:

            if dest.shape[0] == 1:
                blended_img = blended_img[:1]

            blended_img = _norm_helper(blended_img / 255, *self.norm_args, True)

        elif norm_scheme is not None:
            if norm_scheme == '0-1':
                blended_img = (blended_img / 255).astype(init_dtype)
            elif norm_scheme == '0-255':
                blended_img = blended_img.astype(init_dtype)
            elif norm_scheme == 'imagenet':
                blended_img = _norm_helper(blended_img / 255, ['png-r', 'png-g', 'png-b'], None, None, True)
            else:
                assert False, f'Somehow got invalid norm scheme? : {norm_scheme}'

        return blended_img, patch_mask


class PoissonPatchBlender(PatchBlender):

    def __init__(self, mixed_gradients=False):
        self.mix_gradients = mixed_gradients

    def blend(self, factor: float, patch: np.ndarray, patch_mask: np.ndarray, patch_corner: np.ndarray,
              dest: np.ndarray, patch_object_mask: Optional[np.ndarray], dest_object_mask: Optional[np.ndarray]) \
            -> Tuple[np.ndarray, np.ndarray]:
        interp_mask = factor * patch_mask
        blended_img = blend_dst_numpy(dest, patch, interp_mask, patch_corner, mix_gradients=self.mix_gradients,
                                      channels_dim=0)

        return blended_img, patch_mask
