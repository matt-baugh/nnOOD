from multiprocessing import Pool, Queue, Process
from pathlib import Path
import shutil
from typing import List, Optional, Union

import numpy as np
import torch

from nnood.inference.export_utils import save_data_as_file
from nnood.inference.model_restore import load_model_and_checkpoint_files
from nnood.preprocessing.preprocessing import resample_data
from nnood.training.dataloading.dataset_loading import load_npy_or_npz
from nnood.training.network_training.nnOODTrainer import nnOODTrainer
from nnood.utils.file_operations import load_pickle
from nnood.utils.miscellaneous import get_sample_ids_and_files


def predict_from_folder(model: Path, input_folder_path: Path, output_folder_path: Path, folds: Union[List[str], List[int]],
                        save_npz: bool, num_threads_preprocessing: int, num_threads_nifti_save: int,
                        lowres_scores: Optional[Path], part_id: int, num_parts: int, tta: bool,
                        mixed_precision: bool = True, overwrite_existing: bool = True,
                        overwrite_all_in_gpu: bool = None, step_size: float = 0.5,
                        checkpoint_name: str = 'model_final_checkpoint', export_kwargs: dict = None):
    """
    :param model:
    :param input_folder_path:
    :param output_folder_path:
    :param folds:
    :param save_npz:
    :param num_threads_preprocessing:
    :param num_threads_nifti_save:
    :param lowres_scores:
    :param part_id:
    :param num_parts:
    :param tta:
    :param mixed_precision:
    :param overwrite_existing: if not None then it will be overwritten with whatever is in there.
    :param overwrite_all_in_gpu:
    :param step_size:
    :param checkpoint_name:
    :param export_kwargs:
    :return:
    """

    assert input_folder_path.is_dir(), f'Input folder path is not real directory: {input_folder_path}'
    output_folder_path.mkdir(parents=True, exist_ok=True)

    plans_path = Path(model, 'plans.pkl')
    shutil.copy(plans_path, output_folder_path)

    assert plans_path.is_file(), 'Folder with saved model weights must contain a plans.pkl file'
    expected_modalities = load_pickle(plans_path)['dataset_properties']['modalities']

    # check input folder integrity
    sample_id_to_files = get_sample_ids_and_files(input_folder_path, expected_modalities)
    sample_ids = [s_id for (s_id, _) in sample_id_to_files]

    if lowres_scores is not None:
        assert lowres_scores.is_dir(), 'If lowres_scores is not None then it must point to a directory'

        missing_lowres = False
        for sample_id, _ in sample_id_to_files:
            if not (lowres_scores / f'{sample_id}.npy').is_file() and\
               not (lowres_scores / f'{sample_id}.npz').is_file():
                print('Missing file for sample: ', sample_id)
                missing_lowres = True

        assert not missing_lowres, 'Provide lowres scores for missing files listed above.'

    if overwrite_all_in_gpu is None:
        all_in_gpu = False
    else:
        all_in_gpu = overwrite_all_in_gpu

    # Messy but oh well
    input_file_suffix = '.png' if '.png' == sample_id_to_files[0][1][0].suffix else '.nii.gz'

    return predict_cases(model, sample_ids[part_id::num_parts], input_folder_path, input_file_suffix,
                         output_folder_path, folds, save_npz, num_threads_preprocessing, num_threads_nifti_save,
                         lowres_scores, tta, mixed_precision=mixed_precision, overwrite_existing=overwrite_existing,
                         all_in_gpu=all_in_gpu,
                         step_size=step_size, checkpoint_name=checkpoint_name,
                         export_kwargs=export_kwargs)


def predict_cases(model: Path, sample_ids: List[str], input_folder_path: Path, input_file_suffix: str,
                  output_folder_path: Path, folds, save_npz, num_threads_preprocessing, num_threads_nifti_save,
                  scores_from_prev_stage: Optional[Path] = None, do_tta=True, mixed_precision=True,
                  overwrite_existing=False, all_in_gpu=False, step_size=0.5, checkpoint_name='model_final_checkpoint',
                  export_kwargs: dict = None):
    """
    :param model: folder where the model is saved, must contain fold_x subfolders
    :param sample_ids: List of sample id's
    :param input_folder_path: Path for folder containing images for samples
    :param input_file_suffix: Suffix of input files, either .png or .nii.gz
    :param output_folder_path: Directory for output files to be put in
    :param folds: default: (0, 1, 2, 3, 4) (but can also be 'all' or a subset of the five folds, for example use (0, )
    for using only fold_0
    :param save_npz: default: False
    :param num_threads_preprocessing:
    :param num_threads_nifti_save:
    :param scores_from_prev_stage:
    :param do_tta: default: True, can be set to False for a 8x speedup at the cost of a reduced output quality
    :param mixed_precision: if None then we take no action. If True/False we overwrite what the model has in its init
    :param overwrite_existing: default: True
    :param all_in_gpu:
    :param step_size:
    :param checkpoint_name:
    :param export_kwargs:
    :return:
    """
    if scores_from_prev_stage is not None:
        assert scores_from_prev_stage.is_dir()

    if not overwrite_existing:
        print('Number of cases:', len(sample_ids))
        # if save_npz=True then we should also check for missing npz files

        def missing_output_file(s_id: str):
            found_file = False
            found_npz = False

            for f in output_folder_path.iterdir():

                if f.name == (s_id + input_file_suffix):
                    found_file = True
                    # If we don't care about the npz, or we've already found it, then nothing is missing so return False
                    if not save_npz or found_npz:
                        return False

                if save_npz and f.name == (s_id + '.npz'):
                    found_npz = True
                    if found_file:
                        return False

            return True

        sample_ids = list(filter(missing_output_file, sample_ids))

        print('Number of cases that still need to be predicted:', len(sample_ids))

    if len(sample_ids) == 0:
        print('No samples to predict, so skipping rest of prediction process')
        return

    print('Emptying cuda cache')
    torch.cuda.empty_cache()

    print('Loading parameters for folds,', folds)
    trainer, params = load_model_and_checkpoint_files(model, folds, mixed_precision=mixed_precision,
                                                      checkpoint_name=checkpoint_name)
    assert isinstance(trainer, nnOODTrainer)

    if export_kwargs is None:
        if 'export_params' in trainer.plans.keys():
            force_separate_z = trainer.plans['export_params']['force_separate_z']
            interpolation_order = trainer.plans['export_params']['interpolation_order']
            interpolation_order_z = trainer.plans['export_params']['interpolation_order_z']
        else:
            # Same as parameters used for preprocessing resampling, as our scores are continuous values.
            force_separate_z = None
            interpolation_order = 3
            interpolation_order_z = 0
    else:
        force_separate_z = export_kwargs['force_separate_z']
        interpolation_order = export_kwargs['interpolation_order']
        interpolation_order_z = export_kwargs['interpolation_order_z']

    print('Starting preprocessing generator')
    preprocessing = preprocess_multithreaded(trainer, sample_ids, input_folder_path, output_folder_path,
                                             num_threads_preprocessing, scores_from_prev_stage)
    print('Starting prediction...')

    with Pool(num_threads_nifti_save) as pool:
        results = []

        for preprocessed in preprocessing:
            sample_id, (data, sample_properties) = preprocessed
            if isinstance(data, Path):
                real_data = np.load(data)
                data.unlink()
                data = real_data

            print('Predicting', sample_id)
            trainer.load_checkpoint_ram(params[0], False)
            scores = trainer.predict_preprocessed_data(
                data, do_mirroring=do_tta and trainer.data_aug_params['do_mirror'],
                mirror_axes=trainer.data_aug_params['mirror_axes'], use_sliding_window=True, step_size=step_size,
                use_gaussian=True, all_in_gpu=all_in_gpu, mixed_precision=mixed_precision)

            for p in params[1:]:
                trainer.load_checkpoint_ram(p, False)
                scores += trainer.predict_preprocessed_data(
                    data, do_mirroring=do_tta and trainer.data_aug_params['do_mirror'],
                    mirror_axes=trainer.data_aug_params['mirror_axes'], use_sliding_window=True, step_size=step_size,
                    use_gaussian=True, all_in_gpu=all_in_gpu, mixed_precision=mixed_precision)

            if len(params) > 1:
                scores /= len(params)

            transpose_forward = trainer.plans.get('transpose_forward')
            if transpose_forward is not None:
                transpose_backward = trainer.plans.get('transpose_backward')
                scores = scores.transpose([0] + [i + 1 for i in transpose_backward])

            if save_npz:
                npz_file = output_folder_path / f'{sample_id}.npz'
            else:
                npz_file = None

            '''There is a problem with python process communication that prevents us from communicating obejcts 
            larger than 2 GB between processes (basically when the length of the pickle string that will be sent is 
            communicated by the multiprocessing.Pipe object then the placeholder ( I think) does not allow for long 
            enough strings (lol). This could be fixed by changing i to l (for long) but that would require manually 
            patching system python code. We circumvent that problem here by saving softmax_pred to a npy file that will 
            then be read (and finally deleted) by the Process. save_segmentation_nifti_from_softmax can take either 
            filename or np.ndarray and will handle this automatically'''
            bytes_per_voxel = 4
            if all_in_gpu:
                bytes_per_voxel = 2  # if all_in_gpu then the return value is half (float16)
            if np.prod(scores.shape) > (2e9 / bytes_per_voxel * 0.85):  # * 0.85 just to be save
                print('This output is too large for python process-process communication. Saving output temporarily to'
                      'disk')
                temp_file = output_folder_path / f'{sample_id}.npy'
                np.save(temp_file, scores)
                scores = temp_file

            results.append(pool.starmap_async(save_data_as_file,
                                              ((scores, output_folder_path / f'{sample_id}{input_file_suffix}',
                                                sample_properties, interpolation_order, None, None, npz_file, None,
                                                force_separate_z, interpolation_order_z),)
                                              ))

        print('inference done. Now waiting for the exporting to finish...')
        _ = [i.get() for i in results]


def preprocess_multithreaded(trainer: nnOODTrainer, sample_ids: List[str], input_folder_path: Path,
                             output_folder_path: Path, num_processes=2, scores_from_prev_stage: Optional[Path] = None):

    num_processes = min(len(sample_ids), num_processes)

    # I'm not sure why everywhere else uses a Pool and here it's a queue + Processes, but it's from the original code
    # so I'm scared to change it
    q = Queue(1)
    processes = []
    for i in range(num_processes):
        pr = Process(target=preprocess_save_to_queue, args=(trainer.preprocess_sample, q,
                                                            sample_ids[i::num_processes],
                                                            input_folder_path,
                                                            output_folder_path,
                                                            scores_from_prev_stage,
                                                            trainer.plans['transpose_forward']))
        pr.start()
        processes.append(pr)

    try:
        end_ctr = 0
        while end_ctr != num_processes:
            item = q.get()
            if item == 'end':
                end_ctr += 1
                continue
            else:
                yield item

    finally:
        for p in processes:
            if p.is_alive():
                p.terminate()  # this should not happen but better safe than sorry right
            p.join()

        q.close()
        
        
def preprocess_save_to_queue(preprocess_fn, q: Queue, sample_ids: List[str], input_folder: Path,
                             output_folder: Path, scores_from_prev_stage: Optional[Path], transpose_forward):

    if scores_from_prev_stage is not None:
        assert scores_from_prev_stage.is_dir(), 'scores_from_prev_stage in preprocess_save_to_queue is not None, but ' \
                                                'not a directory!'

    errors_in = []
    for i, sample_id in enumerate(sample_ids):
        try:
            print('Preprocessing ',  sample_id)
            data, properties = preprocess_fn(input_folder, sample_id)

            if scores_from_prev_stage is not None:

                scores_prev: np.array = load_npy_or_npz(scores_from_prev_stage / f'{sample_id}.npz', 'r')

                # Check to see that shapes match
                assert (np.array(scores_prev.shape) == properties['original_size']).all(),\
                       'image and scores from previous stage don\'t have the same pixel array shape! image: ' \
                       f'{properties["original_size"]}, scores_prev: {scores_prev.shape}'

                scores_prev = scores_prev.transpose(transpose_forward)
                scores_reshaped = resample_data(scores_prev, data.shape[1:])
                data = np.vstack((data, scores_reshaped)).astype(np.float32)

            '''There is a problem with python process communication that prevents us from communicating obejcts 
            larger than 2 GB between processes (basically when the length of the pickle string that will be sent is 
            communicated by the multiprocessing.Pipe object then the placeholder (I think) does not allow for long 
            enough strings (lol). This could be fixed by changing i to l (for long) but that would require manually 
            patching system python code. We circumvent that problem here by saving softmax_pred to a npy file that will 
            then be read (and finally deleted) by the Process. save_segmentation_nifti_from_softmax can take either 
            filename or np.ndarray and will handle this automatically'''
            print(data.shape)
            if np.prod(data.shape) > (2e9 / 4 * 0.85):  # *0.85 just to be save, 4 because float32 is 4 bytes
                print('This output is too large for python process-process communication. Saving output temporarily to '
                      'disk')
                output_file_path = output_folder / f'{sample_id}.npy'
                np.save(output_file_path, data)
                data = output_file_path
            q.put((sample_id, (data, properties)))
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except Exception as e:
            print('Error in', sample_id)
            print(e)
    q.put('end')
    if len(errors_in) > 0:
        print('There were some errors in the following cases:', errors_in)
        print('These cases were ignored.')
    else:
        print('This worker has ended successfully, no errors to report')
