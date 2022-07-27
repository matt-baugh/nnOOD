from typing import Optional, Tuple, Union

import numpy as np

from nnood.self_supervised_task.patch_transforms.base_transform import PatchTransform


class AdjustContrast(PatchTransform):

    def __init__(self, contrast: Union[float, Tuple[float, float]], min_overlap_pct: Optional[float] = None):
        super().__init__(min_overlap_pct)
        if isinstance(contrast, (tuple, list)):
            self.contrast_min = max(contrast[0], 0.0)
            self.contrast_max = min(contrast[1], 2.0)
        else:
            self.contrast_min = max(1.0 - contrast, 0.0)
            self.contrast_max = min(1.0 + contrast, 1.0)

    def transform(self, patch: np.ndarray, patch_mask: np.ndarray, patch_corner: np.ndarray, dest: np.ndarray,
                  dest_bbox_lbs: np.ndarray, dest_bbox_ubs: np.ndarray, patch_object_mask: Optional[np.ndarray],
                  dest_object_mask: Optional[np.ndarray]) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:

        contrast_f = np.random.uniform(self.contrast_min, self.contrast_max)
        avg = np.mean(patch, axis=0)[patch_mask == 1].mean()
        new_patch = (contrast_f * patch + (1.0 - contrast_f) * avg) * patch_mask

        return new_patch, patch_mask, patch_corner, patch_object_mask


class PsuedoAdjustBrightness(PatchTransform):

    def __init__(self, brightness: Union[float, Tuple[float, float]], min_dset_val: float,
                 min_overlap_pct: Optional[float] = None):
        super().__init__(min_overlap_pct)
        if isinstance(brightness, (tuple, list)):
            self.brightness_min = max(brightness[0], 0.0)
            self.brightness_max = min(brightness[1], 2.0)
        else:
            self.brightness_min = max(1.0 - brightness, 0.0)
            self.brightness_max = min(1.0 + brightness, 1.0)
        self.min_val = min_dset_val

    def transform(self, patch: np.ndarray, patch_mask: np.ndarray, patch_corner: np.ndarray, dest: np.ndarray,
                  dest_bbox_lbs: np.ndarray, dest_bbox_ubs: np.ndarray, patch_object_mask: Optional[np.ndarray],
                  dest_object_mask: Optional[np.ndarray]) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:

        brightness_f = np.random.uniform(self.brightness_min, self.brightness_max)
        new_patch = (self.min_val + (patch - self.min_val) * brightness_f) * patch_mask

        return new_patch, patch_mask, patch_corner, patch_object_mask
