import numpy as np

from nnood.training.network_training.nnOODTrainer import nnOODTrainer
from nnood.training.loss_functions.deep_supervision import MultipleOutputLoss


class nnOODTrainerDS(nnOODTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ds_loss_weights = None

    def setup_DA_params(self):
        super(nnOODTrainerDS, self).setup_DA_params()
        self.deep_supervision_scales = [[1, 1, 1]] + list(list(i) for i in 1 / np.cumprod(
            np.vstack(self.net_num_pool_op_kernel_sizes), axis=0))[:-1]

    def initialize(self, training=True, force_load_plans=False):
        """
        Add loss function wrapper for deep supervision.

        :param training:
        :param force_load_plans:
        :return:
        """
        if not self.was_initialized:
            super(nnOODTrainerDS, self).initialize(training, force_load_plans)

            # Set up deep supervision loss
            # We need to know the number of outputs of the network
            net_numpool = len(self.net_num_pool_op_kernel_sizes)

            # We give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

            # We don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
            weights[~mask] = 0
            weights = weights / weights.sum()
            self.ds_loss_weights = weights

            self.loss = MultipleOutputLoss(self.loss, self.ds_loss_weights)

    def run_online_evaluation(self, output, target):
        """
        due to deep supervision the return value and the reference are now lists of tensors. We only need the full
        resolution output because this is what we are interested in in the end. The others are ignored
        :param output:
        :param target:
        :return:
        """
        target = target[0]
        output = output[0]
        return super().run_online_evaluation(output, target)

    def _wrap_ds_fn(self, tmp_ds_val: bool, fn, *args, **kwargs):
        """
        Helper for disabling deep supervision when wrapping functions
        :param fn:
        :param args:
        :param kwargs:
        :return:
        """
        ds = self.network.do_ds
        self.network.do_ds = tmp_ds_val

        ret = fn(*args, **kwargs)

        self.network.do_ds = ds

        return ret

    def validate(self, *args, **kwargs):
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction.
        """
        return self._wrap_ds_fn(False, super(nnOODTrainerDS, self).validate, *args, **kwargs)

    def predict_preprocessed_data(self, *args, **kwargs) -> np.ndarray:
        return self._wrap_ds_fn(False, super().predict_preprocessed_data, *args, **kwargs)

    def run_training(self):
        return self._wrap_ds_fn(True, super(nnOODTrainerDS, self).run_training)
