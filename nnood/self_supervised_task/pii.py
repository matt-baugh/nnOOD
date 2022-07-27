import numpy as np
import torch
import torch.nn.functional as F

from nnood.self_supervised_task.patch_blender import PoissonPatchBlender
from nnood.self_supervised_task.patch_ex import patch_ex
from nnood.self_supervised_task.patch_shape_maker import EqualUniformPatchMaker
from nnood.self_supervised_task.patch_labeller import ContinuousPatchLabeller
from nnood.self_supervised_task.self_sup_task import SelfSupTask


class PII(SelfSupTask):

    def __init__(self, mixed_gradients=False):
        self.mixed_gradients = mixed_gradients

        self.shape_maker = EqualUniformPatchMaker()
        self.prev_sample = None
        self.labeller = ContinuousPatchLabeller()
        self.blender = PoissonPatchBlender(self.mixed_gradients)

    def apply(self, sample, sample_mask, sample_properties, sample_fn=None, dest_bbox=None, return_locations=False):

        sample = np.array(sample)

        src = np.array(sample_fn(False)[0]) if self.prev_sample is None else self.prev_sample
        if isinstance(src, tuple):
            src = src[0]

        result = patch_ex(sample, src, shape_maker=self.shape_maker, blender=self.blender, labeller=self.labeller,
                          binary_factor=False, dest_bbox=dest_bbox, extract_within_bbox=True,
                          return_anomaly_locations=return_locations, width_bounds_pct=(0.1, 0.4))
        self.prev_sample = sample
        return result

    def loss(self, pred, target):
        return F.binary_cross_entropy_with_logits(pred, target)

    def label_is_seg(self):
        return False

    def inference_nonlin(self, data):
        return torch.sigmoid(data)
