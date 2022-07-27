import numpy as np
import torch
import torch.nn.functional as F

from nnood.self_supervised_task.patch_blender import UniformPatchBlender
from nnood.self_supervised_task.patch_ex import patch_ex
from nnood.self_supervised_task.patch_shape_maker import UnequalUniformPatchMaker
from nnood.self_supervised_task.patch_labeller import ContinuousPatchLabeller
from nnood.self_supervised_task.self_sup_task import SelfSupTask


class RectFPI(SelfSupTask):

    def __init__(self):
        self.shape_maker = UnequalUniformPatchMaker(sample_dist=self.sample_dimension)
        self.prev_sample = None
        self.blender = UniformPatchBlender()
        self.labeller = ContinuousPatchLabeller()

    @staticmethod
    def sample_dimension(lb, ub, img_d):
        gamma_lb = 0.03
        gamma_shape = 2
        gamma_scale = 0.1

        gamma_sample = (gamma_lb + np.random.gamma(gamma_shape, gamma_scale)) * img_d

        return int(np.clip(gamma_sample, lb, ub))

    def apply(self, sample, sample_mask, sample_properties, sample_fn=None, dest_bbox=None, return_locations=False):
        src = sample_fn(False)[0] if self.prev_sample is None else self.prev_sample
        if isinstance(src, tuple):
            src = src[0]

        result = patch_ex(sample, src, shape_maker=self.shape_maker, blender=self.blender, labeller=self.labeller,
                          binary_factor=False, dest_bbox=dest_bbox, extract_within_bbox=True,
                          return_anomaly_locations=return_locations, width_bounds_pct=(0.06, 0.8))
        self.prev_sample = sample
        return result

    def loss(self, pred, target):
        return F.binary_cross_entropy_with_logits(pred, target)

    def label_is_seg(self):
        return False

    def inference_nonlin(self, data):
        return torch.sigmoid(data)
