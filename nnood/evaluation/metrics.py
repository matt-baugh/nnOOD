from typing import Union, List
import bisect

import cv2
import numpy as np
from sklearn import metrics


def _flatten_all(predictions: Union[List[np.ndarray], np.ndarray], labels: Union[List[np.ndarray], np.ndarray]):
    assert type(predictions) == type(labels), f'Types of predictions and labels in evaluation metric must match: ' \
                                              f'{type(predictions)}, {type(labels)}'
    if type(predictions) is list:
        predictions = np.concatenate([p.flatten() for p in predictions])
        labels = np.concatenate([l.flatten() for l in labels])
    else:
        predictions = predictions.flatten()
        labels = labels.flatten()

    assert len(predictions) == len(labels), f'Length of predictions and labels in evaluation metric must match: ' \
                                            f'{len(predictions)}, {len(labels)}'
    return predictions, labels


def auroc(predictions: Union[List[np.ndarray], np.ndarray], labels: Union[List[np.ndarray], np.ndarray], **kwargs):
    predictions, labels = _flatten_all(predictions, labels)
    if kwargs.get('return_curve', False):
        fpr, tpr, _ = metrics.roc_curve(labels, predictions)
        return metrics.roc_auc_score(labels, predictions), fpr, tpr
    else:
        return metrics.roc_auc_score(labels, predictions)


def average_precision(predictions: Union[List[np.ndarray], np.ndarray], labels: Union[List[np.ndarray], np.ndarray],
                      **kwargs):
    predictions, labels = _flatten_all(predictions, labels)
    if kwargs.get('return_curve', False):
        precision, recall, _ = metrics.precision_recall_curve(labels, predictions)
        # Don't use auc function, as that uses trapezoidal rule which can be too optimistic
        return metrics.average_precision_score(labels, predictions), precision, recall
    else:
        return metrics.average_precision_score(labels, predictions)


def per_region_overlap(predictions: List[np.ndarray], labels: List[np.ndarray], **kwargs):
    max_fpr = kwargs.get('pro_max_fpr', 0.3)
    max_components = kwargs.get('pro_max_components', 25)

    flat_preds, flat_labels = _flatten_all(predictions, labels)
    fpr, _, thresholds = metrics.roc_curve(flat_labels, flat_preds)
    split = len(fpr[fpr < max_fpr])
    # last thresh has fpr >= max_fpr
    fpr = fpr[:(split + 1)]
    thresholds = thresholds[:(split + 1)]
    neg_thresholds = -thresholds
    for p in predictions:
        p[p < thresholds[-1]] = 0

    # calculate per-component-overlap for each threshold and match to global thresholds
    pro = np.zeros_like(fpr)
    total_components = 0
    for j in range(len(labels)):
        num_labels, label_img = cv2.connectedComponents(np.uint8(labels[j]))
        if num_labels > max_components:
            print(f'Invalid label map: too many components ({num_labels}) skipping sample {j}.')
        if num_labels == 1:  # only background
            continue
        total_components += num_labels - 1

        y_score = predictions[j].flatten()
        desc_score_indices = np.argsort(y_score, kind='mergesort')[::-1]
        y_score = y_score[desc_score_indices]
        distinct_value_indices = np.where(np.diff(y_score))[0]
        threshold_idxs = np.r_[distinct_value_indices, y_score.size - 1]
        thresholds_j = y_score[threshold_idxs]
        for k in range(1, num_labels):
            y_true = np.uint8(label_img == k).flatten()
            y_true = y_true[desc_score_indices]
            tps = np.cumsum(y_true)[threshold_idxs]
            tpr = tps / tps[-1]

            # match tprs to global thresholds so that we can calculate pro
            right = len(thresholds)
            for tpr_t, t in zip(tpr[::-1], thresholds_j[::-1]):  # iterate in ascending order
                if t < thresholds[-1]:  # remove too small thresholds
                    continue
                i = bisect.bisect_left(neg_thresholds, -t, hi=right)  # search for negated as thresholds desc
                pro[i: right] += tpr_t
                right = i
    pro /= total_components

    if fpr[-1] > max_fpr:  # interpolate last value
        pro[-1] = ((max_fpr - fpr[-2]) * pro[-1] + (fpr[-1] - max_fpr) * pro[-2]) / (fpr[-1] - fpr[-2])
        fpr[-1] = max_fpr

    if kwargs.get('return_curve', False):
        return metrics.auc(fpr, pro) / max_fpr, fpr, pro
    else:
        return metrics.auc(fpr, pro) / max_fpr


def all_score_stats(predictions: List[np.ndarray], labels: List[np.ndarray], **kwargs):
    predictions, _ = _flatten_all(predictions, labels)
    return np.mean(predictions, dtype=float), np.std(predictions, dtype=float)


def anomaly_score_stats(predictions: List[np.ndarray], labels: List[np.ndarray], **kwargs):
    predictions, labels = _flatten_all(predictions, labels)
    anomaly_scores = predictions[labels.astype(bool)]
    return np.mean(anomaly_scores, dtype=float), np.std(anomaly_scores, dtype=float)


def normal_score_stats(predictions: List[np.ndarray], labels: List[np.ndarray], **kwargs):
    predictions, labels = _flatten_all(predictions, labels)
    normal_scores = predictions[np.invert(labels.astype(bool))]
    return np.mean(normal_scores, dtype=float), np.std(normal_scores, dtype=float)


ALL_METRICS = {
    'AUROC': auroc,
    'AP score': average_precision,
    #    'AU-PRO': per_region_overlap,   # Uses OpenCV so only works on 2D, uint8 images.
}
