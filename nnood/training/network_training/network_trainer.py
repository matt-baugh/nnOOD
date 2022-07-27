import os
import sys
from abc import abstractmethod
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from time import time, sleep
from typing import Optional, Union
from warnings import warn

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from sklearn.model_selection import KFold
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import lr_scheduler
from tqdm import trange

from nnood.network_architecture.neural_network import AnomalyScoreNetwork
from nnood.utils.file_operations import load_pickle, save_pickle
from nnood.utils.to_torch import maybe_to_torch, to_cuda

matplotlib.use('agg')


class NetworkTrainer:
    def __init__(self, deterministic=True, fp16=False):
        """
        A generic class that can train almost any neural network (RNNs excluded). It provides basic functionality such
        as the training loop, tracking of training and validation losses (and the target metric if you implement it)
        Training can be terminated early if the validation loss (or the target metric if implemented) do not improve
        anymore. This is based on a moving average (MA) of the loss/metric instead of the raw values to get more smooth
        results.
        What you need to override:
        - __init__
        - initialize
        - run_online_evaluation (optional)
        - finish_online_evaluation (optional)
        - validate
        - predict_test_case
        """
        self.fp16 = fp16
        self.amp_grad_scaler = None
        self.no_print = False

        if deterministic:
            np.random.seed(12345)
            torch.manual_seed(12345)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(12345)
            cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True

        # SET THESE IN self.initialize() ###############################################################################
        self.network: Optional[Union[AnomalyScoreNetwork, nn.DataParallel]] = None
        self.optimizer = None
        self.lr_scheduler = None
        self.tr_gen = self.val_gen = None
        self.was_initialized = False

        # SET THESE IN INIT ############################################################################################
        self.output_folder: Optional[Path] = None
        self.fold = None
        self.loss = None
        self.dataset_directory: Optional[Path] = None

        # SET THESE IN LOAD_DATASET OR DO_SPLIT ########################################################################

        # dataset can be None for inference mode
        self.dataset = None
        # dataset_trlr do not need to be used, they just appear if you are using the suggested load_dataset_and_do_split
        self.dataset_tr = self.dataset_val = None

        # THESE DO NOT NECESSARILY NEED TO BE MODIFIED #################################################################
        self.patience = 125
        self.val_eval_criterion_alpha = 0.9  # alpha * old + (1-alpha) * new
        # if this is too low then the moving average will be too noisy and the training may terminate early. If it is
        # too high the training will take forever
        self.train_loss_MA_alpha = 0.9  # alpha * old + (1-alpha) * new
        self.train_loss_MA_eps = 5e-4  # new MA must be at least this much better (smaller)
        self.max_num_epochs = 2500
        self.num_batches_per_epoch = 100
        self.num_val_batches_per_epoch = 20
        self.also_val_in_tr_mode = False
        self.lr_threshold = 1e-6  # Currently unused, previously in nnU-Net prevented early stopping for lr above this

        self.val_eval_criterion_threshold = 0.875  # Stop training when reach this performance on validation set

        # LEAVE THESE ALONE ############################################################################################
        self.val_eval_criterion_MA = None
        self.train_loss_MA = None
        self.best_val_eval_criterion_MA = None
        self.best_MA_tr_loss_for_patience = None
        self.best_epoch_based_on_MA_tr_loss = None
        self.all_tr_losses = []
        self.all_val_losses = []
        self.all_val_losses_tr_mode = []
        self.all_val_eval_metrics = []  # does not have to be used
        self.epoch = 0
        self.log_file = None
        self.deterministic = deterministic

        self.use_progress_bar = False
        if 'nnood_use_progress_bar' in os.environ.keys():
            self.use_progress_bar = bool(int(os.environ['nnood_use_progress_bar']))

        # Settings for saving checkpoints ##############################################################################
        self.save_every = 50
        self.save_latest_only = True  # if false it will not store/overwrite _latest but separate files each
        # time an intermediate checkpoint is created
        self.save_intermediate_checkpoints = True  # whether or not to save checkpoint_latest
        self.save_best_checkpoint = True  # save the best checkpoint according to self.best_val_eval_criterion_MA
        self.save_final_checkpoint = True  # whether or not to save the final checkpoint

    @abstractmethod
    def initialize(self, training=True):
        """
        create self.output_folder
        modify self.output_folder if you are doing cross-validation (one folder per fold)
        set self.tr_gen and self.val_gen
        call self.initialize_network and self.initialize_optimizer_and_scheduler (important!)
        finally set self.was_initialized to True
        :param training:
        :return:
        """

    @abstractmethod
    def load_dataset(self):
        pass

    def do_split(self):
        if self.fold == 'all':
            # if fold==all then we use all images for training and validation
            tr_keys = val_keys = list(self.dataset.keys())
        else:
            splits_file = self.dataset_directory / 'splits_final.pkl'

            # if the split file does not exist we need to create it
            if not splits_file.is_file():
                self.print_to_log_file('Creating new 5-fold cross-validation split...')
                splits = []
                all_keys_sorted = np.sort(list(self.dataset.keys()))
                kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
                for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
                    train_keys = np.array(all_keys_sorted)[train_idx]
                    test_keys = np.array(all_keys_sorted)[test_idx]
                    splits.append(OrderedDict())
                    splits[-1]['train'] = train_keys
                    splits[-1]['val'] = test_keys
                save_pickle(splits, splits_file)

            else:
                self.print_to_log_file('Using splits from existing split file:', splits_file)
                splits = load_pickle(splits_file)
                self.print_to_log_file('The split file contains %d splits.' % len(splits))

            self.print_to_log_file(f'Desired fold for training: {self.fold}')
            if self.fold < len(splits):
                tr_keys = splits[self.fold]['train']
                val_keys = splits[self.fold]['val']
                self.print_to_log_file('This split has %d training and %d validation cases.'
                                       % (len(tr_keys), len(val_keys)))
            else:
                self.print_to_log_file('INFO: You requested fold %d for training but splits '
                                       'contain only %d folds. I am now creating a '
                                       'random (but seeded) 80:20 split!' % (self.fold, len(splits)))
                # if we request a fold that is not in the split file, create a random 80:20 split
                rnd = np.random.RandomState(seed=12345 + self.fold)
                keys = np.sort(list(self.dataset.keys()))
                idx_tr = rnd.choice(len(keys), int(len(keys) * 0.8), replace=False)
                idx_val = [i for i in range(len(keys)) if i not in idx_tr]
                tr_keys = [keys[i] for i in idx_tr]
                val_keys = [keys[i] for i in idx_val]
                self.print_to_log_file('This random 80:20 split has %d training and %d validation cases.'
                                       % (len(tr_keys), len(val_keys)))

        tr_keys.sort()
        val_keys.sort()
        self.dataset_tr = OrderedDict()
        for i in tr_keys:
            self.dataset_tr[i] = self.dataset[i]
        self.dataset_val = OrderedDict()
        for i in val_keys:
            self.dataset_val[i] = self.dataset[i]

    def plot_progress(self):
        """
        Should probably by improved
        :return:
        """
        try:
            font = {'weight': 'normal',
                    'size': 18}

            matplotlib.rc('font', **font)

            fig = plt.figure(figsize=(30, 24))
            ax = fig.add_subplot(111)
            ax2 = ax.twinx()

            x_values = list(range(self.epoch + 1))

            ax.plot(x_values, self.all_tr_losses, color='b', ls='-', label='loss_tr')

            ax.plot(x_values, self.all_val_losses, color='r', ls='-', label='loss_val, train=False')

            if len(self.all_val_losses_tr_mode) > 0:
                ax.plot(x_values, self.all_val_losses_tr_mode, color='g', ls='-', label='loss_val, train=True')
            if len(self.all_val_eval_metrics) == len(x_values):
                ax2.plot(x_values, self.all_val_eval_metrics, color='g', ls='--', label='evaluation metric')

            ax.set_xlabel('epoch')
            ax.set_ylabel('loss')
            ax2.set_ylabel('evaluation metric')
            ax.legend()
            ax2.legend(loc=9)

            fig.savefig(self.output_folder / 'progress.png')
            plt.close()
        except IOError:
            self.print_to_log_file('failed to plot: ', sys.exc_info())

    def print_to_log_file(self, *args, also_print_to_console=True, add_timestamp=True):

        if not self.no_print:
            timestamp = time()
            dt_object = datetime.fromtimestamp(timestamp)

            if add_timestamp:
                args = ('%s:' % dt_object, *args)

            if self.log_file is None:
                self.output_folder.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now()
                self.log_file = self.output_folder / ('training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt' %
                                                      (timestamp.year, timestamp.month, timestamp.day, timestamp.hour,
                                                       timestamp.minute, timestamp.second))
                with open(self.log_file, 'w') as f:
                    f.write('Starting... \n')
            successful = False
            max_attempts = 5
            ctr = 0
            while not successful and ctr < max_attempts:
                try:
                    with open(self.log_file, 'a+') as f:
                        for a in args:
                            f.write(str(a))
                            f.write(' ')
                        f.write('\n')
                    successful = True
                except IOError:
                    print('%s: failed to log: ' % datetime.fromtimestamp(timestamp), sys.exc_info())
                    sleep(0.5)
                    ctr += 1
            if also_print_to_console:
                print(*args)

    def save_checkpoint(self, file_path: Path, save_optimizer=True):
        start_time = time()
        state_dict = self.network.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()
        lr_sched_state_dct = None
        if self.lr_scheduler is not None and hasattr(self.lr_scheduler,
                                                     'state_dict'):
            lr_sched_state_dct = self.lr_scheduler.state_dict()
        if save_optimizer:
            optimizer_state_dict = self.optimizer.state_dict()
        else:
            optimizer_state_dict = None

        self.print_to_log_file('saving checkpoint...')
        save_this = {
            'epoch': self.epoch + 1,
            'state_dict': state_dict,
            'optimizer_state_dict': optimizer_state_dict,
            'lr_scheduler_state_dict': lr_sched_state_dct,
            'plot_stuff': (self.all_tr_losses, self.all_val_losses, self.all_val_losses_tr_mode,
                           self.all_val_eval_metrics),
            'best_stuff': (
                self.best_epoch_based_on_MA_tr_loss, self.best_MA_tr_loss_for_patience,
                self.best_val_eval_criterion_MA)}
        if self.amp_grad_scaler is not None:
            save_this['amp_grad_scaler'] = self.amp_grad_scaler.state_dict()

        torch.save(save_this, file_path)
        self.print_to_log_file('done, saving took %.2f seconds' % (time() - start_time))

    def load_best_checkpoint(self, train=True):
        if self.fold is None:
            raise RuntimeError('Cannot load best checkpoint if self.fold is None')

        best_model_file = self.output_folder / 'model_best.model'
        if best_model_file.is_file():
            self.load_checkpoint(best_model_file, train=train)
        else:
            self.print_to_log_file('WARNING! model_best.model does not exist! Cannot load best checkpoint. Falling '
                                   'back to load_latest_checkpoint')
            self.load_latest_checkpoint(train)

    def load_latest_checkpoint(self, train=True):
        possible_model_files = ['model_final_checkpoint.model', 'model_latest.model', 'model_best.model']

        for p_m_f in possible_model_files:
            p_m_f = self.output_folder / p_m_f
            if p_m_f.is_file():
                if p_m_f.name == 'model_best.model':
                    return self.load_best_checkpoint(train)
                else:
                    return self.load_checkpoint(p_m_f, train=train)

        raise RuntimeError('No checkpoint found')

    def load_final_checkpoint(self, train=False):
        filename = self.output_folder / 'model_final_checkpoint.model'
        if not filename.is_file():
            raise RuntimeError('Final checkpoint not found. Expected: %s. Please finish the training first.' % filename)
        return self.load_checkpoint(filename, train=train)

    def load_checkpoint(self, fname, train=True):
        self.print_to_log_file('loading checkpoint', fname, 'train=', train)
        if not self.was_initialized:
            self.initialize(train)
        saved_model = torch.load(fname, map_location=torch.device('cpu'))
        self.load_checkpoint_ram(saved_model, train)

    @abstractmethod
    def initialize_network(self):
        """
        initialize self.network here
        :return:
        """
        pass

    @abstractmethod
    def initialize_optimizer_and_scheduler(self):
        """
        initialize self.optimizer and self.lr_scheduler (if applicable) here
        :return:
        """
        pass

    def load_checkpoint_ram(self, checkpoint, train=True):
        """
        used for if the checkpoint is already in ram
        :param checkpoint:
        :param train:
        :return:
        """
        if not self.was_initialized:
            self.initialize(train)

        new_state_dict = OrderedDict()
        curr_state_dict_keys = list(self.network.state_dict().keys())
        # if state dict comes form nn.DataParallel but we use non-parallel model here then the state dict keys do not
        # match. Use heuristic to make it match
        for k, value in checkpoint['state_dict'].items():
            key = k
            if key not in curr_state_dict_keys and key.startswith('module.'):
                key = key[7:]
            new_state_dict[key] = value

        if self.fp16:
            self._maybe_init_amp()
            if train:
                if 'amp_grad_scaler' in checkpoint.keys():
                    self.amp_grad_scaler.load_state_dict(checkpoint['amp_grad_scaler'])

        self.network.load_state_dict(new_state_dict)
        self.epoch = checkpoint['epoch']
        if train:
            optimizer_state_dict = checkpoint['optimizer_state_dict']
            if optimizer_state_dict is not None:
                self.optimizer.load_state_dict(optimizer_state_dict)

            if self.lr_scheduler is not None and hasattr(self.lr_scheduler, 'load_state_dict') and \
                    checkpoint['lr_scheduler_state_dict'] is not None:
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

            if issubclass(self.lr_scheduler.__class__, lr_scheduler._LRScheduler):
                self.lr_scheduler.step(self.epoch)

        self.all_tr_losses, self.all_val_losses, self.all_val_losses_tr_mode, self.all_val_eval_metrics = checkpoint[
            'plot_stuff']

        # load best loss (if present)
        if 'best_stuff' in checkpoint.keys():
            self.best_epoch_based_on_MA_tr_loss, self.best_MA_tr_loss_for_patience, self.best_val_eval_criterion_MA = \
                checkpoint['best_stuff']

        self._maybe_init_amp()

    def _maybe_init_amp(self):
        if self.fp16 and self.amp_grad_scaler is None:
            self.amp_grad_scaler = GradScaler()

    def plot_network_architecture(self):
        """
        can be implemented (see nnOODTrainer) but does not have to. Not implemented here because it imposes stronger
        assumptions on the presence of class variables
        :return:
        """
        pass

    def run_training(self):
        if not torch.cuda.is_available():
            self.print_to_log_file('WARNING!!! You are attempting to run training on a CPU (torch.cuda.is_available() '
                                   'is False). This can be VERY slow!')

        _ = self.tr_gen.next()
        _ = self.val_gen.next()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._maybe_init_amp()

        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.plot_network_architecture()

        if cudnn.benchmark and cudnn.deterministic:
            warn('torch.backends.cudnn.deterministic is True indicating a deterministic training is desired. '
                 'But torch.backends.cudnn.benchmark is True as well and this will prevent deterministic training! '
                 'If you want deterministic then set benchmark=False')

        if not self.was_initialized:
            self.initialize(True)

        while self.epoch < self.max_num_epochs:
            self.print_to_log_file('\nepoch: ', self.epoch)
            epoch_start_time = time()
            train_losses_epoch = []

            # train one epoch
            self.network.train()

            if self.use_progress_bar:
                with trange(self.num_batches_per_epoch) as t_bar:
                    for _ in t_bar:
                        t_bar.set_description('Epoch {}/{}'.format(self.epoch + 1, self.max_num_epochs))

                        curr_loss = self.run_iteration(self.tr_gen, True)

                        t_bar.set_postfix(loss=curr_loss)
                        train_losses_epoch.append(curr_loss)
            else:
                for _ in range(self.num_batches_per_epoch):
                    curr_loss = self.run_iteration(self.tr_gen, True)
                    train_losses_epoch.append(curr_loss)

            self.all_tr_losses.append(np.mean(train_losses_epoch))
            self.print_to_log_file('train loss : %.4f' % self.all_tr_losses[-1])

            with torch.no_grad():
                # validation with train=False
                self.network.eval()
                val_losses = []
                for b in range(self.num_val_batches_per_epoch):
                    curr_loss = self.run_iteration(self.val_gen, False, True)
                    val_losses.append(curr_loss)
                self.all_val_losses.append(np.mean(val_losses))
                self.print_to_log_file('validation loss: %.4f' % self.all_val_losses[-1])

                if self.also_val_in_tr_mode:
                    self.network.train()
                    # validation with train=True
                    val_losses = []
                    for b in range(self.num_val_batches_per_epoch):
                        curr_loss = self.run_iteration(self.val_gen, False)
                        val_losses.append(curr_loss)
                    self.all_val_losses_tr_mode.append(np.mean(val_losses))
                    self.print_to_log_file('validation loss (train=True): %.4f' % self.all_val_losses_tr_mode[-1])

            self.update_train_loss_MA()  # needed for lr scheduler and stopping of training

            continue_training = self.on_epoch_end()

            epoch_end_time = time()

            if not continue_training:
                # allows for early stopping
                break

            self.epoch += 1
            self.print_to_log_file('This epoch took %f s\n' % (epoch_end_time - epoch_start_time))

        self.epoch -= 1  # if we don't do this we can get a problem with loading model_final_checkpoint.

        if self.save_final_checkpoint:
            self.save_checkpoint(self.output_folder / 'model_final_checkpoint.model')
        # now we can delete latest as it will be identical with final
        latest_model_file = self.output_folder / 'model_latest.model'
        latest_model_pkl_file = self.output_folder / 'model_latest.pkl'

        if latest_model_file.is_file():
            os.remove(latest_model_file)
        if latest_model_pkl_file.is_file():
            os.remove(latest_model_pkl_file)

    def maybe_update_lr(self):
        # maybe update learning rate
        if self.lr_scheduler is not None:
            assert isinstance(self.lr_scheduler, (lr_scheduler.ReduceLROnPlateau, lr_scheduler._LRScheduler))

            if isinstance(self.lr_scheduler, lr_scheduler.ReduceLROnPlateau):
                # lr scheduler is updated with moving average val loss. should be more robust
                self.lr_scheduler.step(self.train_loss_MA)
            else:
                self.lr_scheduler.step(self.epoch + 1)
        self.print_to_log_file('lr is now (scheduler) %s' % str(self.optimizer.param_groups[0]['lr']))

    def maybe_save_checkpoint(self):
        """
        Saves a checkpoint every save_ever epochs.
        :return:
        """
        if self.save_intermediate_checkpoints and (self.epoch % self.save_every == (self.save_every - 1)):
            self.print_to_log_file('saving scheduled checkpoint file...')
            if not self.save_latest_only:
                self.save_checkpoint(self.output_folder / ('model_ep_%03.0d.model' % (self.epoch + 1)))
            self.save_checkpoint(self.output_folder / 'model_latest.model')
            self.print_to_log_file('done')

    def update_eval_criterion_MA(self):
        """
        If self.all_val_eval_metrics is unused (len=0) then we fall back to using -self.all_val_losses for the MA to
        determine early stopping (not a minimization, but a maximization of a metric and therefore the - in the latter
        case)
        :return:
        """
        if self.val_eval_criterion_MA is None:
            if len(self.all_val_eval_metrics) == 0:
                self.val_eval_criterion_MA = - self.all_val_losses[-1]
            else:
                self.val_eval_criterion_MA = self.all_val_eval_metrics[-1][0]
        else:
            if len(self.all_val_eval_metrics) == 0:
                '''
                We here use alpha * old - (1 - alpha) * new because new in this case is the validation loss and lower
                is better, so we need to negate it.
                '''
                self.val_eval_criterion_MA = self.val_eval_criterion_alpha * self.val_eval_criterion_MA - (
                        1 - self.val_eval_criterion_alpha) * self.all_val_losses[-1]
            else:
                self.val_eval_criterion_MA = self.val_eval_criterion_alpha * self.val_eval_criterion_MA + (
                        1 - self.val_eval_criterion_alpha) * self.all_val_eval_metrics[-1][0]

        self.print_to_log_file('Current validation criterion moving average: ', self.val_eval_criterion_MA)

    def manage_patience(self):
        continue_training = True
        if self.patience is not None:
            # if best_MA_tr_loss_for_patience and best_epoch_based_on_MA_tr_loss were not yet initialized,
            # initialize them
            if self.best_MA_tr_loss_for_patience is None:
                self.best_MA_tr_loss_for_patience = self.train_loss_MA

            if self.best_epoch_based_on_MA_tr_loss is None:
                self.best_epoch_based_on_MA_tr_loss = self.epoch

            if self.best_val_eval_criterion_MA is None:
                self.best_val_eval_criterion_MA = self.val_eval_criterion_MA

            # check if the current epoch is the best one according to moving average of validation criterion. If so
            # then save 'best' model
            # Do not use this for validation. This is intended for test set prediction only.

            if self.val_eval_criterion_MA > self.best_val_eval_criterion_MA:
                self.best_val_eval_criterion_MA = self.val_eval_criterion_MA
                # self.print_to_log_file('saving best epoch checkpoint...')
                if self.save_best_checkpoint:
                    self.save_checkpoint(self.output_folder / 'model_best.model')

            # Keep training if we're below the threshold
            continue_training = self.val_eval_criterion_MA < self.val_eval_criterion_threshold

            # Stop if reach threshold, or loss plateaus, whichever first.

            # Now see if the moving average of the train loss has improved. If yes then reset patience, else
            # increase patience
            if self.train_loss_MA + self.train_loss_MA_eps < self.best_MA_tr_loss_for_patience:
                self.best_MA_tr_loss_for_patience = self.train_loss_MA
                self.best_epoch_based_on_MA_tr_loss = self.epoch

            # if patience has reached its maximum then finish training (provided lr is low enough)
            if self.epoch - self.best_epoch_based_on_MA_tr_loss > self.patience:
                if self.optimizer.param_groups[0]['lr'] > self.lr_threshold:
                    self.best_epoch_based_on_MA_tr_loss = self.epoch - self.patience // 2
                else:
                    continue_training = False

        return continue_training

    def on_epoch_end(self):
        # does not have to do anything, but can be used to update self.all_val_eval_metrics
        self.finish_online_evaluation()

        self.plot_progress()

        self.maybe_update_lr()

        self.maybe_save_checkpoint()

        self.update_eval_criterion_MA()

        continue_training = self.manage_patience()
        return continue_training

    def update_train_loss_MA(self):
        if self.train_loss_MA is None:
            self.train_loss_MA = self.all_tr_losses[-1]
        else:
            self.train_loss_MA = self.train_loss_MA_alpha * self.train_loss_MA + (1 - self.train_loss_MA_alpha) * \
                                 self.all_tr_losses[-1]

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        data_dict = next(data_generator)

        data = maybe_to_torch(data_dict['data'])
        target = maybe_to_torch(data_dict['target'])

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

        self.optimizer.zero_grad()

        if self.fp16:
            with autocast():
                output = self.network(data)
                del data
                curr_loss = self.loss(output, target)
                if torch.any(torch.isnan(curr_loss)):
                    # TODO: to YIKES or not to YIKES?
                    print('YIKES')

            if do_backprop:
                self.amp_grad_scaler.scale(curr_loss).backward()
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            output = self.network(data)
            del data
            curr_loss = self.loss(output, target)

            if do_backprop:
                curr_loss.backward()
                self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target

        return curr_loss.detach().cpu().numpy()

    def run_online_evaluation(self, *args, **kwargs):
        """
        Can be implemented, does not have to
        :param output_torch:
        :param target_npy:
        :return:
        """
        pass

    def finish_online_evaluation(self):
        """
        Can be implemented, does not have to
        :return:
        """
        pass

    @abstractmethod
    def validate(self, *args, **kwargs):
        pass
