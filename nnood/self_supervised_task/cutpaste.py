from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple

from nnood.configuration import default_num_processes
from nnood.training.dataloading.dataset_loading import load_npy_or_npz
from nnood.self_supervised_task.patch_blender import SwapPatchBlender
from nnood.self_supervised_task.patch_ex import patch_ex
from nnood.self_supervised_task.patch_shape_maker import UnequalUniformPatchMaker
from nnood.self_supervised_task.patch_labeller import BinaryPatchLabeller
from nnood.self_supervised_task.patch_transforms.colour_transforms import AdjustContrast, PsuedoAdjustBrightness
from nnood.self_supervised_task.patch_transforms.spatial_transforms import TranslatePatch
from nnood.self_supervised_task.self_sup_task import SelfSupTask


def cutpaste_sample_dimensions(_: List[Tuple[int, int]], img_dims: np.ndarray):
    num_dimensions = len(img_dims)

    patch_area = np.random.uniform(0.02, 0.15) * np.product(img_dims)
    dim_aspect_ratios = [np.random.choice([np.random.uniform(0.3, 1), np.random.uniform(1, 3.3)])
                         for _ in range(num_dimensions - 1)]

    area_root = np.power(patch_area, 1 / num_dimensions)

    shape = []
    for i in range(num_dimensions):
        # First is multplied by sqrt(AR[0])
        # Last is multiplied by sqrt(1 / AR[-1])
        # All others are multiplied by sqrt(AR[i] / AR[i - 1])
        # Causes product(shape) = Area

        num = 1.0 if i == num_dimensions - 1 else dim_aspect_ratios[i]
        den = 1.0 if i == 0 else dim_aspect_ratios[i - 1]

        shape.append(np.round(area_root * np.sqrt(num / den)).astype(int))

    # Shuffle dimensions, to avoid early/later dimensions being more extreme than middle ones
    np.random.shuffle(shape)

    return shape


def _load_get_min(f):
    return load_npy_or_npz(f, 'r').min()


class CutPaste(SelfSupTask):

    def __init__(self):
        self.shape_maker = UnequalUniformPatchMaker(sample_dist=cutpaste_sample_dimensions, calc_dims_together=True)
        self.transformations = [AdjustContrast(0.1), TranslatePatch()]
        self.blender = SwapPatchBlender()
        self.labeller = BinaryPatchLabeller()
        self.calibrated = False

    def calibrate(self, dataset, exp_plans):

        if not self.calibrated:
            print('Calibrating CutPaste...')
            dataset_min_val = np.inf

            files_to_load = [v['data_file'] for v in dataset.values()]

            with Pool(default_num_processes) as p:
                mins = p.map(_load_get_min, files_to_load)

            dataset_min_val = np.minimum(dataset_min_val, np.amin(mins))

            assert dataset_min_val != np.inf, 'Minimum dataset value np.inf, is the dataset empty?'

            self.transformations = [PsuedoAdjustBrightness(0.1, dataset_min_val)] + self.transformations
            self.calibrated = True
        else:
            print('WARNING: CutPaste has already been calibrated, cannot be done again.')

    def apply(self, sample, sample_mask, sample_properties, sample_fn=None, dest_bbox=None, return_locations=False):
        # Note: Width_bounds_pct aren't used to sample dimensions (as CutPaste chooses them based on patch area and
        # aspect ratios, meaning they only serve to decide how close to the edge we put patches to the edge. UB is
        # meaningless

        if not self.calibrated:
            print('WARNING: CutPaste has not been calibrated, so cannot use PseudoBrightness transform')

        result = patch_ex(sample, same=True, shape_maker=self.shape_maker, patch_transforms=self.transformations,
                          blender=self.blender, labeller=self.labeller, binary_factor=True, dest_bbox=dest_bbox,
                          return_anomaly_locations=return_locations, width_bounds_pct=(0.15, 0.30))
        return result

    def loss(self, pred, target):
        return F.binary_cross_entropy_with_logits(pred, target)

    def label_is_seg(self):
        return False

    def inference_nonlin(self, data):
        return torch.sigmoid(data)


if __name__ == '__main__':
    np.seterr(all='raise')
    test_input = np.random.random((3, 50, 50))
    test_output = CutPaste()(test_input)[0]

    assert not (test_input == test_output).all()
