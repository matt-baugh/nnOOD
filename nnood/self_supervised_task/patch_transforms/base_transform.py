from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np


class PatchTransform(ABC):

    def __init__(self, min_overlap_pct: Optional[float] = None):
        self.min_overlap_pct = min_overlap_pct

    @abstractmethod
    def transform(self, patch: np.ndarray, patch_mask: np.ndarray, patch_corner: np.ndarray, dest: np.ndarray,
                  dest_bbox_lbs: np.ndarray, dest_bbox_ubs: np.ndarray, patch_object_mask: Optional[np.ndarray],
                  dest_object_mask: Optional[np.ndarray]) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        :param patch: Patch extracted from source image.
        :param patch_mask: Mask of which elements of patch will be put/blended into the destination image.
        :param patch_corner: Coordinate of the minimum element of patch relative to the destination image.
        :param dest: Destination image, available for querying but probably shouldn't change!
        :param dest_bbox_lbs: Lower bound of box to be extracted from destination image
        :param dest_bbox_ubs: Upper bound of box to be extracted from destination image
        :param patch_object_mask: Mask showing which elements of the patch contain an object of interest.
        :param dest_object_mask: Mask showing which elements of the destination image contain an object of interest.
        :returns Tuple containing updated patch, patch_mask, patch_corner and patch_object_mask
        """
        pass

    def __call__(self, patch: np.ndarray, patch_mask: np.ndarray, patch_corner: np.ndarray, dest: np.ndarray,
                 dest_bbox_lbs: np.ndarray, dest_bbox_ubs: np.ndarray, patch_object_mask: Optional[np.ndarray],
                 dest_object_mask: Optional[np.ndarray]) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        return self.transform(patch, patch_mask, patch_corner, dest, dest_bbox_lbs, dest_bbox_ubs, patch_object_mask,
                              dest_object_mask)
