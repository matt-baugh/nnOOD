from abc import ABC, abstractmethod

import numpy as np
from scipy.ndimage import binary_closing, grey_closing


class PatchLabeller(ABC):

    def __init__(self, tolerance=0.01):
        self.tolerance = tolerance

    @abstractmethod
    def label(self, factor: float, blended_img: np.ndarray, orig_img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        :param factor: Extent that patch is used in blending. In range [0-1]. May be ignored.
        :param blended_img: Image with patches blended within it.
        :param orig_img: Original image, prior to blending.
        :param mask: Mask of where patches have been blended into blended_img.
        """
        pass

    def __call__(self, factor: float, blended_img: np.ndarray, orig_img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return self.label(factor, blended_img, orig_img, mask)

    def remove_no_change(self, blended_img: np.ndarray, orig_img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        mask = (np.mean(mask * np.abs(blended_img - orig_img), axis=0) > self.tolerance).astype(int)
        # Remove grain from threshold choice, using scipy morphology
        # Equivalent to using structure of (5, 5, 5)
        return binary_closing(mask, structure=np.ones([3] * len(mask.shape)), iterations=2)


class BinaryPatchLabeller(PatchLabeller):

    def label(self, factor: float, blended_img: np.ndarray, orig_img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return self.remove_no_change(blended_img, orig_img, mask)


class ContinuousPatchLabeller(PatchLabeller):

    def label(self, factor: float, blended_img: np.ndarray, orig_img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return factor * self.remove_no_change(blended_img, orig_img, mask)


class IntensityPatchLabeller(PatchLabeller):

    def label(self, factor: float, blended_img: np.ndarray, orig_img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        mask = self.remove_no_change(blended_img, orig_img, mask)

        label = np.mean(mask * np.abs(blended_img - orig_img), axis=0)
        return grey_closing(label, size=[7] * len(mask.shape))


class LogisticIntensityPatchLabeller(IntensityPatchLabeller):

    def __init__(self, k, x0, tolerance=0.01):
        super().__init__(tolerance)
        self.k = k
        self.x0 = x0

    def label(self, factor: float, blended_img: np.ndarray, orig_img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        intensity_label = super().label(factor, blended_img, orig_img, mask)
        return (intensity_label > 0).astype(int) / (1 + np.exp(-self.k * (intensity_label - self.x0)))
