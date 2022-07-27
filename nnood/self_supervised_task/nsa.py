from multiprocessing import Pool

import numpy as np
import torch
from torch.nn import functional as F

from nnood.configuration import default_num_processes
from nnood.self_supervised_task.nsa_utils import nsa_sample_dimension, compute_nsa_mask_params
from nnood.self_supervised_task.patch_labeller import IntensityPatchLabeller, LogisticIntensityPatchLabeller
from nnood.self_supervised_task.patch_blender import PoissonPatchBlender
from nnood.self_supervised_task.patch_ex import patch_ex
from nnood.self_supervised_task.patch_shape_maker import UnequalUniformPatchMaker
from nnood.self_supervised_task.patch_transforms.spatial_transforms import ResizePatch, TranslatePatch
from nnood.self_supervised_task.self_sup_task import SelfSupTask
from nnood.training.dataloading.dataset_loading import load_npy_or_npz


class NSA(SelfSupTask):
    def __init__(self, mix_gradients=False):
        self.mix_gradients = mix_gradients
        self.min_obj_overlap_pct = 0.25
        self.shape_maker = UnequalUniformPatchMaker(sample_dist=nsa_sample_dimension)
        self.transforms = [ResizePatch(min_overlap_pct=0, validate_position=False),
                           TranslatePatch(min_overlap_pct=self.min_obj_overlap_pct)]
        self.blender = PoissonPatchBlender(self.mix_gradients)

        self.prev_sample = self.prev_sample_mask = None

        self._calibrated = False
        self.width_bounds_pct = self.labeller = self.min_obj_pct = self.class_has_foreground = self.num_patches = None

    def apply(self, sample, sample_mask, sample_properties, sample_fn=None, dest_bbox=None, return_locations=False):
        src = sample_fn(sample_mask is not None)[0] if self.prev_sample is None else self.prev_sample

        if sample_mask is not None:
            if self.prev_sample is None:
                src_mask = src[1]
                src = src[0]
            else:
                src_mask = self.prev_sample_mask
        else:
            src_mask = None

        assert self._calibrated, 'NSA task requires calibration!'

        result = patch_ex(sample, src,
                          shape_maker=self.shape_maker,
                          patch_transforms=self.transforms,
                          blender=self.blender,
                          labeller=self.labeller,
                          binary_factor=True,
                          # 0 overlap for initial patch source location, as we check that when translating patch.
                          min_overlap_pct=0.0,
                          width_bounds_pct=self.width_bounds_pct,
                          min_object_pct=self.min_obj_pct if self.class_has_foreground else None,
                          dest_bbox=dest_bbox,
                          num_patches=self.num_patches,
                          skip_background=(sample_mask, src_mask) if self.class_has_foreground else None,
                          return_anomaly_locations=return_locations)

        self.prev_sample = sample
        self.prev_sample_mask = sample_mask
        return result

    def _load_img_m_pair(self, f):
        curr_img = load_npy_or_npz(f, 'r', self.class_has_foreground)

        if self.class_has_foreground:
            return curr_img  # Actually a tuple of image and mask
        else:
            return curr_img, None

    def _collect_continuous_NSA_examples(self, files_to_load):
        last_img, last_img_m = self._load_img_m_pair(files_to_load[0]['data_file'])

        patch_changes = []
        temp_labeller = IntensityPatchLabeller()

        for j in files_to_load[1:]:
            new_img, new_img_m = self._load_img_m_pair(j['data_file'])

            _, cont_label = patch_ex(new_img, last_img,
                                     shape_maker=self.shape_maker,
                                     patch_transforms=self.transforms,
                                     blender=self.blender,
                                     labeller=temp_labeller,
                                     binary_factor=True,
                                     min_overlap_pct=0.0,
                                     width_bounds_pct=self.width_bounds_pct,
                                     min_object_pct=self.min_obj_pct,
                                     num_patches=self.num_patches,
                                     skip_background=(new_img_m, last_img_m)
                                     if self.class_has_foreground else None)

            patch_changes.append(cont_label[cont_label > 0])

            last_img = new_img
            last_img_m = new_img_m

        return np.concatenate(patch_changes)

    def calibrate(self, dataset, exp_plans):
        if not self._calibrated:

            data_num_dims = len(exp_plans['transpose_forward'])
            self.class_has_foreground = exp_plans['dataset_properties']['has_uniform_background']

            # Compute NSA parameters based on the object masks
            self.width_bounds_pct, self.num_patches, self.min_obj_pct = \
                compute_nsa_mask_params(self.class_has_foreground, dataset, data_num_dims)

            # Measure distribution of changes caused by NSA anomalies
            keys = list(dataset.keys())

            with Pool(default_num_processes) as pool:
                num_test_samples = 500
                samples_per_process = num_test_samples // default_num_processes

                all_patch_changes = pool.map(self._collect_continuous_NSA_examples,
                                             [[dataset[keys[j % len(keys)]]
                                               for j in range(i, i + samples_per_process)]
                                              for i in range(0, num_test_samples, samples_per_process)])

            all_patch_changes = np.concatenate(all_patch_changes)

            # Calculate logistic function parameters such that:
            # - lower bound of patch labels is 0.1.
            # - patches saturate at 40th percentile of changes observed.

            scale = np.log(99 * 9) / np.percentile(all_patch_changes, 40)
            x0 = np.log(9) / scale
            self.labeller = LogisticIntensityPatchLabeller(scale, x0)

            self._calibrated = True
        else:
            print('WARNING: NSA has already been calibrated, cannot be done again.')

    def loss(self, pred, target):
        return F.binary_cross_entropy_with_logits(pred, target)

    def label_is_seg(self):
        return False

    def inference_nonlin(self, data):
        return torch.sigmoid(data)


class NSAMixed(NSA):

    def __init__(self):
        super().__init__(mix_gradients=True)
