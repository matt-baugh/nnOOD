from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
from collections import OrderedDict


class SelfSupTask(ABC):
    """
    A generic class defining the form of a self-supervised anomaly detection task.
    What you need to override:
    - __apply__
    """

    def calibrate(self, dataset, exp_plans):
        """
        If any parameters of the task depend on the dataset being used, set them here (for example, whether data is 2D
        or 3D). May not be needed.
        :param dataset: dictionary of all training/validation samples, so use dataset_loading.load_npy_or_npz to
                actually get samples
        :param exp_plans: plans for experiment
        """
        pass

    @abstractmethod
    def apply(self, sample: np.ndarray, sample_mask: Optional[np.ndarray], sample_properties: OrderedDict,
              sample_fn: Optional[
                  Callable[[bool], Tuple[Union[Tuple[np.ndarray, np.ndarray], np.ndarray], Any]]] = None,
              dest_bbox: Optional[np.ndarray] = None, return_locations: bool = False) \
            -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, List[np.ndarray]]]:
        """
        Apply the self-supervised task to the single data sample.
        :param sample:
        :param sample_mask: Optional mask showing region of interest (allows task to focus on making anomalies here).
        :param sample_properties:
        :param sample_fn: Optional function to get auxiliary samples for task (such as in FPI)
        :param dest_bbox: Specify which region to apply the anomalies within. If None, treat as entire image.
        :param return_locations: Optional boolean for whether to return list of anomaly centres
        :return: sample with task applied, label map and (if return_locations=True) a list of anomaly centres.
        """
        pass

    def __call__(self, sample: np.ndarray, sample_mask: Optional[np.ndarray], sample_properties: OrderedDict,
                 sample_fn: Optional[
                     Callable[[bool], Tuple[Union[Tuple[np.ndarray, np.ndarray], np.ndarray], Any]]] = None,
                 dest_bbox: Optional[np.ndarray] = None, return_locations: bool = False) \
            -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]]:
        return self.apply(sample, sample_mask, sample_properties, sample_fn, dest_bbox, return_locations)

    @abstractmethod
    def loss(self, pred, target):
        """
        Loss function to be used when training with this task. May be simply calling a torch loss function
        :param pred:
        :param target:
        :return:
        """

    @abstractmethod
    def label_is_seg(self):
        """
        Returns whether the label for this loss function is a segmentation (of classes) or not.
        :return:
        """

    def inference_nonlin(self, data):
        """
        Optional nonlinearity to be applied to network output at inference time.
        :param data:
        :return:
        """
        return data
