from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np


class PatchShapeMaker(ABC):

    @abstractmethod
    def get_patch_mask(self, dim_bounds: List[Tuple[int, int]], img_dims: np.ndarray) -> np.ndarray:
        """
        :param dim_bounds: Tuples giving lower and upper bounds for patch size in each dimension
        :param img_dims: Image dimensions, can be used as scaling factor.
        Creates a patch mask to be used in the self-supervised task.
        Mask must have length(dim_bounds) dimensions.
        """
        pass

    def __call__(self, dim_bounds: List[Tuple[int, int]], img_dims: np.ndarray) -> np.ndarray:
        return self.get_patch_mask(dim_bounds, img_dims)


# For squares, cubes, etc
class EqualUniformPatchMaker(PatchShapeMaker):

    def __init__(self, sample_dist=lambda lb, ub, _: np.random.randint(lb, ub)):
        self.sample_dist = sample_dist

    def get_patch_mask(self, dim_bounds: List[Tuple[int, int]], img_dims: np.ndarray) -> np.ndarray:
        lbs, ubs = zip(*dim_bounds)

        # As all dimensions must be equal, take maximum lower bound and minimum upperbound as patch size bounds
        # and give the minimum image dimension to be used as an optional scaling factor
        patch_dim = self.sample_dist(max(lbs), min(ubs), min(img_dims))
        return np.ones([patch_dim] * len(dim_bounds), dtype=bool)


# For rectangles, cuboids, etc
class UnequalUniformPatchMaker(PatchShapeMaker):

    def __init__(self, sample_dist=lambda lb, ub, _: np.random.randint(lb, ub), calc_dims_together=False):
        self.sample_dist = sample_dist
        self.calc_dims_together = calc_dims_together

    def get_patch_mask(self, dim_bounds: List[Tuple[int, int]], img_dims: np.ndarray) -> np.ndarray:

        shape = self.sample_dist(dim_bounds, img_dims) if self.calc_dims_together else\
            [self.sample_dist(lb, ub, d) for ((lb, ub), d) in zip(dim_bounds, img_dims)]
        return np.ones(shape, dtype=bool)

