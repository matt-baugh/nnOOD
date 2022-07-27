import argparse

from nnood.paths import default_plans_identifier
from nnood.training.network_training.nnOODTrainer import nnOODTrainer
from nnood.utils.default_configuration import get_default_configuration
from nnood.utils.miscellaneous import load_pretrained_weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('network', help='can only be one of: \'lowres\', \'fullres\', \'cascade_fullres\'')
    parser.add_argument('network_trainer', help='class name of trainer to be used')
    parser.add_argument('dataset', help='Dataset Name')
    parser.add_argument('task', help='Task name')
    parser.add_argument('fold', help='0, 1, ..., 5 or \'all\'')
    parser.add_argument('-val', '--validation_only', help='use this if you want to only run the validation',
                        action='store_true')
    parser.add_argument('-c', '--continue_training', help='use this if you want to continue a training',
                        action='store_true')
    parser.add_argument('-cf', '--continue_from',
                        help='Use this to specify which checkpoint to continue training from. Must also set'
                             '--continue_training. Must be one of [model_final_checkpoint, model_latest, model_best].'
                             'Optional, if not set then training continues from latest checkpoint.',
                        required=False, default=None)
    parser.add_argument('-p', help='plans identifier. Only change this if you created a custom experiment planner',
                        default=default_plans_identifier, required=False)
    parser.add_argument('--use_compressed_data', default=False, action='store_true',
                        help='If you set use_compressed_data, the training cases will not be decompressed. Reading '
                             'compressed data is much more CPU and RAM intensive and should only be used if you know '
                             'what you are '
                             'doing', required=False)
    parser.add_argument('--deterministic',
                        help='Makes training deterministic, but reduces training speed substantially. Probably not '
                             'necessary. Deterministic training will make you overfit to some random seed. Don\'t use '
                             'that.',
                        required=False, default=False, action='store_true')
    parser.add_argument('--npz', required=False, default=False, action='store_true', help='if set then nnood will '
                                                                                          'export npz files of '
                                                                                          'predicted segmentations '
                                                                                          'in the validation as well. '
                                                                                          'This is needed to run the '
                                                                                          'ensembling step so unless '
                                                                                          'you are developing nnUNet '
                                                                                          'you should enable this')
    parser.add_argument('--fp32', required=False, default=False, action='store_true',
                        help='disable mixed precision training and run old school fp32')
    parser.add_argument('--val_folder', required=False, default='validation_raw',
                        help='name of the validation folder. No need to use this for most people')
    parser.add_argument('--disable_saving', required=False, action='store_true',
                        help='If set nnU-Net will not save any parameter files (except a temporary checkpoint that '
                             'will be removed at the end of the training). Useful for development when you are '
                             'only interested in the results and want to save some disk space')
    parser.add_argument('--val_disable_overwrite', action='store_false', default=True,
                        help='Validation does not overwrite existing segmentations')
    parser.add_argument('-pretrained_weights', type=str, required=False, default=None,
                        help='path to nnU-Net checkpoint file to be used as pretrained model (use .model file, for '
                             'example model_final_checkpoint.model). Will only be used when actually training. '
                             'Optional. Beta. Use with caution.')
    parser.add_argument('--load_dataset_ram', required=False, action='store_true', default=False,
                        help='Load entire dataset into RAM, use carefully, only suggested for smaller, 2D datasets.')

    args = parser.parse_args()

    dataset = args.dataset
    task = args.task
    fold = args.fold
    network = args.network
    network_trainer = args.network_trainer
    validation_only = args.validation_only
    plans_identifier = args.p

    continue_training = args.continue_training
    continue_from = args.continue_from
    load_dataset_ram = args.load_dataset_ram

    use_compressed_data = args.use_compressed_data
    decompress_data = not use_compressed_data

    deterministic = args.deterministic

    fp32 = args.fp32
    run_mixed_precision = not fp32

    val_folder = args.val_folder

    if fold == 'all':
        pass
    else:
        fold = int(fold)

    plans_file, output_folder_name, dataset_directory, stage, trainer_class, task_class =\
        get_default_configuration(network, dataset, task, network_trainer, plans_identifier)

    if trainer_class is None:
        raise RuntimeError('Could not find trainer class in nnood.training.network_training')
    if task_class is None:
        raise RuntimeError('Could not find task class in nnood.training.network_training')

    if network == 'cascade_fullres':
        assert False, 'Trying to run cascade full res, but I haven\'t made that yet!'
        # assert issubclass(trainer_class, (nnUNetTrainerCascadeFullRes, nnUNetTrainerV2CascadeFullRes)), \
        #     'If running 3d_cascade_fullres then your ' \
        #     'trainer class must be derived from ' \
        #     'nnOODTrainerCascadeFullRes'
    else:
        assert issubclass(trainer_class,
                          nnOODTrainer), 'network_trainer was found but is not derived from nnOODTrainer'

    trainer = trainer_class(plans_file, fold, task_class, output_folder=output_folder_name,
                            dataset_directory=dataset_directory, stage=stage, unpack_data=decompress_data,
                            deterministic=deterministic, fp16=run_mixed_precision, load_dataset_ram=load_dataset_ram)
    if args.disable_saving:
        trainer.save_final_checkpoint = False  # whether or not to save the final checkpoint
        trainer.save_best_checkpoint = False   # whether or not to save the best checkpoint according to
        trainer.save_intermediate_checkpoints = True  # whether or not to save checkpoint_latest. We need that in case
        # the training crashes
        trainer.save_latest_only = True  # if false it will not store/overwrite _latest but separate files each

    trainer.initialize(not validation_only)

    if not validation_only:
        if continue_training:
            # -c was set, continue a previous training and ignore pretrained weights
            if continue_from:
                assert continue_from in ['model_final_checkpoint', 'model_latest', 'model_best'],\
                    f'Unexpected checkpoint name: {continue_from}'

                checkpoint_file = trainer.output_folder / f'{continue_from}.model'
                assert checkpoint_file.is_file(), f'Missing checkpoint file: {checkpoint_file}'
                trainer.load_checkpoint(checkpoint_file)
            else:
                trainer.load_latest_checkpoint()

        elif (not continue_training) and (args.pretrained_weights is not None):
            # Start a new training, using pre-trained weights.
            load_pretrained_weights(trainer.network, args.pretrained_weights)
        else:
            # new training without pretrained weights, do nothing
            pass

        trainer.run_training()
    else:
        trainer.load_final_checkpoint(train=False)

    trainer.network.eval()

    # Predict validation
    # trainer.validate(save=args.npz, validation_folder_name=val_folder, overwrite=args.val_disable_overwrite)

    if network == 'lowres':
        print('FYI, even though this is a lowres experiment, we don\'t predict next stage as data in self-sup tasks '
              'is dynamic.')


if __name__ == '__main__':
    main()
