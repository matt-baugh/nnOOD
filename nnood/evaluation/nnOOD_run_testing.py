import argparse
from pathlib import Path

import numpy as np
import torch

from nnood.evaluation.nnOOD_evaluate_folder import evaluate_folder
from nnood.inference.predict import predict_from_folder
from nnood.paths import default_plans_identifier, preprocessed_data_base, raw_data_base, results_base
from nnood.utils.file_operations import load_pickle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', help='Dataset which the model was trained on.', required=True)
    parser.add_argument('-t', '--task_name', help='Self-supervised task which the model was trained on', required=True)
    parser.add_argument('-tr', '--trainer_class_name',
                        help='Name of the nnOODTrainer used for full resolution and low resolution U-Net. If you are '
                             'running inference with the cascade and the folder pointed to by --lowres_maps '
                             'does not contain the anomaly maps generated by the low resolution U-Net then the low '
                             'resolution anomaly maps will be automatically generated. For this case, make sure to set '
                             'the trainer class here that matches your --cascade_trainer_class_name.',
                        required=True)
    parser.add_argument('-ctr', '--cascade_trainer_class_name',
                        help='Trainer class name used for predicting the full resolution U-Net part of the cascade.',
                        default=None, required=False)
    parser.add_argument('-id', '--test_identifier', default=None, required=False,
                        help='Use identifier when making results folder, optional.')
    parser.add_argument('-p', '--plans_identifier', help='do not touch this unless you know what you are doing',
                        default=default_plans_identifier, required=False)
    parser.add_argument('-f', '--folds', nargs='+', default='None',
                        help='folds to use for prediction. Default is None which means that folds will be detected '
                             'automatically in the model output folder')
    parser.add_argument('--num_threads_preprocessing', required=False, default=6, type=int,
                        help='Determines many background processes will be used for data preprocessing. Reduce this if '
                             'you run into out of memory (RAM) problems. Default: 6')
    parser.add_argument('--num_threads_save', required=False, default=2, type=int,
                        help='Determines many background processes will be used for exporting. Reduce this if you run '
                             'into out of memory (RAM) problems. Default: 2')
    parser.add_argument('--disable_tta', required=False, default=False, action='store_true',
                        help='set this flag to disable test time data augmentation via mirroring. Speeds up inference '
                             'by roughly factor 4 (2D) or 8 (3D)')
    parser.add_argument('--overwrite_existing', required=False, default=False, action='store_true',
                        help='Set this flag if the target folder contains predictions that you would like to overwrite')
    parser.add_argument('--all_in_gpu', type=str, default='None', required=False, help='can be None, False or True. '
                                                                                       'Do not touch.')
    parser.add_argument('--step_size', type=float, default=0.5, required=False, help='don\'t touch')
    parser.add_argument('-chk',
                        help='checkpoint name, default: model_final_checkpoint',
                        required=False,
                        default='model_final_checkpoint')
    parser.add_argument('--disable_mixed_precision', default=False, action='store_true', required=False,
                        help='Predictions are done with mixed precision by default. This improves speed and reduces '
                             'the required vram. If you want to disable mixed precision you can set this flag. Note '
                             'that this is not recommended (mixed precision is ~2x faster!)')
    parser.add_argument('--lowres_only', required=False, default=False, action='store_true',
                        help='Set this flag if you want to only use the lowres stage of a 2 step pipeline')

    args = parser.parse_args()
    task_name = args.task_name
    dataset = args.dataset
    plans_identifier = args.plans_identifier
    folds = args.folds
    num_threads_preprocessing = args.num_threads_preprocessing
    num_threads_save = args.num_threads_save
    disable_tta = args.disable_tta
    step_size = args.step_size
    overwrite_existing = args.overwrite_existing
    all_in_gpu = args.all_in_gpu
    trainer_class_name = args.trainer_class_name
    cascade_trainer_class_name = args.cascade_trainer_class_name
    lowres_only = args.lowres_only

    if isinstance(folds, list):
        if folds[0] == 'all' and len(folds) == 1:
            pass
        else:
            folds = [int(i) for i in folds]
    elif folds == 'None':
        folds = None
    else:
        raise ValueError('Unexpected value for argument folds')

    assert all_in_gpu in ['None', 'False', 'True']
    if all_in_gpu == 'None':
        all_in_gpu = None
    elif all_in_gpu == 'True':
        all_in_gpu = True
    elif all_in_gpu == 'False':
        all_in_gpu = False

    plans_file = Path(preprocessed_data_base, dataset, plans_identifier)
    assert plans_file.is_file(), f'Missing plans file: {plans_file}'

    input_folder = Path(raw_data_base, dataset, 'imagesTs')
    labels_folder = Path(raw_data_base, dataset, 'labelsTs')
    assert input_folder.is_dir(), f'Missing test images folder: {input_folder}'
    assert labels_folder.is_dir(), f'Missing test labels folder: {labels_folder}'

    plans = load_pickle(plans_file)
    possible_stages = list(plans['plans_per_stage'].keys())

    models = ['fullres'] if len(possible_stages) == 1 else ['lowres', 'cascade_fullres']

    if lowres_only:
        assert 'lowres' in models, 'Cannot run lowres only on a pipeline without a lowres stage!'
        models = ['lowres']

    if 'cascade_fullres' in models:
        assert cascade_trainer_class_name is not None, 'Cannot use cascade_fullres model without defining' \
                                                       'cascade_trainer_class_name'

    output_folder_base = Path(results_base, dataset, task_name, 'testResults', plans_identifier)
    if args.test_identifier is not None:
        output_folder_base /= args.test_identifier

    lowres_scores = None

    for model in models:
        print(f'Starting predictions for {model}')

        curr_trainer = cascade_trainer_class_name if model == 'cascade_fullres' else trainer_class_name
        curr_output = output_folder_base / ('lowres_predictions' if model == 'lowres' and not lowres_only else '')

        if model == 'cascade_fullres':
            assert lowres_scores.is_dir(), 'Somehow attempting cascade_fullres without lowres_scores being a dir.'

        model_folder = Path(results_base, dataset, task_name, model, curr_trainer + '__' + plans_identifier)
        print(f'Model is stored in: {model_folder}')
        assert model_folder.is_dir(), f'Model output folder not found, expected: {model_folder}'

        predict_from_folder(model_folder, input_folder, curr_output, folds, True, num_threads_preprocessing,
                            num_threads_save, lowres_scores, 0, 1, not disable_tta,
                            mixed_precision=not args.disable_mixed_precision, overwrite_existing=overwrite_existing,
                            overwrite_all_in_gpu=all_in_gpu, step_size=step_size, checkpoint_name=args.chk)

        if model == 'lowres':
            lowres_scores = curr_output
            torch.cuda.empty_cache()

    label_suffix = 'png' if np.array(['png' in p for p in plans['modalities'].values()]).any() else 'nii.gz'

    print(f'Starting test evaluation, with label suffix {label_suffix}')
    evaluate_folder(labels_folder.__str__(), output_folder_base.__str__(), label_suffix, 'npz')


if __name__ == '__main__':
    main()
