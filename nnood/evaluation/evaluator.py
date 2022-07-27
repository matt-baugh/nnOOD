from collections import OrderedDict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
from typing import Dict, List, Optional, Tuple

import SimpleITK as sitk
import numpy as np
from tqdm import tqdm

from nnood.evaluation.metrics import ALL_METRICS
from nnood.training.dataloading.dataset_loading import load_npy_or_npz
from nnood.utils.file_operations import save_json

TARGET_METRIC_BATCH_SIZE = 25


def _load_file_pairs_list(pred_ref_pairs: List[Tuple[Path, Optional[Path]]]) -> List[Tuple[np.ndarray, np.ndarray]]:
    def _load_file(f: Path) -> np.ndarray:

        if f.suffix == '.npz' or f.suffix == '.npy':
            # A file saved in a numpy representation must be in the correct form (channels first).
            return load_npy_or_npz(f, 'r')

        sitk_img = sitk.ReadImage(f.__str__())
        img = sitk.GetArrayFromImage(sitk_img)

        sitk_shape = np.array(sitk_img.GetSize())[::-1]
        img_shape = np.array(img.shape)

        if len(img_shape) != len(sitk_shape):
            # Must be a vector image, represented with channels last. Convert to channels first.
            return np.moveaxis(img, -1, 0)
        else:
            # No channels, so no swap needed.
            return img

    def _load_file_pair(p_r_pair: Tuple[Path, Optional[Path]]) -> Tuple[np.ndarray, np.ndarray]:
        p, r = p_r_pair
        pred = _load_file(p)

        # If ref is None, then image is normal, so return array of zeroes.
        if r is None:
            return pred, np.zeros_like(pred)
        else:
            return pred, _load_file(r)

    return [_load_file_pair(p) for p in pred_ref_pairs]


def compute_metric_scores(pred_ref_file_pairs: List[Tuple[Path, Optional[Path]]], **metric_kwargs) -> Dict:
    pred_ref_img_pairs = _load_file_pairs_list(pred_ref_file_pairs)

    preds = [p for p, _ in pred_ref_img_pairs]
    ref_labels = [l for _, l in pred_ref_img_pairs]

    all_ref_labels = np.concatenate([r.flatten() for r in ref_labels])
    label_values = np.unique(all_ref_labels)

    # Evaluation labels should be binary (whether or not anomaly is present)
    unexpected_labels = [v for v in label_values if v not in [0, 1]]
    if len(unexpected_labels) > 0:
        # print('Binarising reference labels as found values other than [0, 1]: ', unexpected_labels)
        for r_l in ref_labels:
            r_l[r_l != 0] = 1

    # Default to computing all metrics
    chosen_metrics = metric_kwargs.get('metrics', ALL_METRICS.keys())
    results = OrderedDict()

    for m in tqdm(chosen_metrics, desc='Computing different metrics...'):
        results[m] = ALL_METRICS[m](preds, ref_labels, **metric_kwargs)

    return results


def aggregate_scores(pred_ref_file_pairs: List[Tuple[Path, Optional[Path]]],
                     json_output_file=None,
                     json_name='',
                     json_description='',
                     json_author='Anonymous',
                     json_task='',
                     **metric_kwargs) -> Dict:
    """
    test = predicted image
    :param pred_ref_file_pairs:
    :param json_output_file:
    :param json_name:
    :param json_description:
    :param json_author:
    :param json_task:
    :param metric_kwargs:
    :return:
    """

    random.shuffle(pred_ref_file_pairs)

    num_pairs = len(pred_ref_file_pairs)
    if num_pairs < TARGET_METRIC_BATCH_SIZE:
        print('Computing metrics over entire dataset')
        results = compute_metric_scores(pred_ref_file_pairs, **metric_kwargs)
    else:
        num_batches = round(num_pairs / TARGET_METRIC_BATCH_SIZE)
        batch_size = int(num_pairs / num_batches)

        print(f'Computing metrics over {num_batches} batches, each of size around {batch_size}')

        all_results = []
        last_index = 0
        for i in range(num_batches - 1):
            print(f'Computing batch {i}')
            last_index = (i + 1) * batch_size
            all_results.append(compute_metric_scores(pred_ref_file_pairs[i * batch_size: last_index], **metric_kwargs))

        print('Computing final batch')
        all_results.append(compute_metric_scores(pred_ref_file_pairs[last_index: num_pairs], **metric_kwargs))

        results = {}
        for k in all_results[0].keys():
            results[k] = np.mean([r[k] for r in all_results])

    # We create a hopefully unique id by hashing the entire output dictionary
    if json_output_file is not None:
        json_dict = OrderedDict()
        json_dict['name'] = json_name
        json_dict['description'] = json_description
        timestamp = datetime.today()
        json_dict['timestamp'] = str(timestamp)
        json_dict['task'] = json_task
        json_dict['author'] = json_author
        json_dict['results'] = results
        json_dict['id'] = hashlib.md5(json.dumps(json_dict).encode('utf-8')).hexdigest()[:12]
        save_json(json_dict, json_output_file)

    return results
