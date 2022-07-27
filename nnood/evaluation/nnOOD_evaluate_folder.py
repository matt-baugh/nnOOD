from pathlib import Path
from typing import Optional

from nnood.evaluation.evaluator import aggregate_scores


def evaluate_folder(folder_with_gts: str, folder_with_predictions: str, ref_suffix: str, pred_suffix: str,
                    **metric_kwargs):
    """
    writes a summary.json to folder_with_predictions
    :param folder_with_gts: folder where the ground truth segmentations are saved. Must be nifti files.
    :param folder_with_predictions: folder where the predicted segmentations are saved. Must be nifti files.
    :param ref_suffix:
    :param pred_suffix:
    :return:
    """
    folder_with_gts = Path(folder_with_gts)
    folder_with_predictions = Path(folder_with_predictions)
    # Can't just check suffix as same suffix's include '.' within them (like .nii.gz) and pathlib only counts the final
    # .XXX as the suffix
    files_pred = [f for f in folder_with_predictions.iterdir() if f.is_file() and f.name.endswith(pred_suffix)]

    files_gt = [folder_with_gts / (f.name[:-len(pred_suffix)] + ref_suffix) for f in files_pred]

    missing_gts = []

    def check_gt_exists(f: Path) -> Optional[Path]:
        if f.is_file():
            return f
        else:
            missing_gts.append(f.name)
            return None

    files_gt = list(map(check_gt_exists, files_gt))

    # noinspection PySimplifyBooleanCheck
    if missing_gts != []:
        print(f'Files missing gt, assumed to be entirely normal ({len(missing_gts)} in total).')
        for f in missing_gts:
            print(f)

    test_ref_pairs = list(zip(files_pred, files_gt))
    res = aggregate_scores(test_ref_pairs, json_output_file=folder_with_predictions / 'summary.json',
                           **metric_kwargs)
    print()
    print('Evaluation results:')
    print(res)


def main():
    import argparse
    parser = argparse.ArgumentParser('Evaluates the anomaly scores located in the folder pred. Output of this script '
                                     'is a json file. At the very bottom of the json file is going to be a \'mean\' '
                                     'entry with averages metrics across all cases.')
    parser.add_argument('-ref', required=True, type=str, help='Folder containing the reference labels.')
    parser.add_argument('-pred', required=True, type=str, help='Folder containing the predicted scores. File names '
                                                               'must match between the folders!')
    parser.add_argument('-r_s', '--ref_suffix', required=True, type=str, help='File suffix of the reference images.')
    parser.add_argument('-p_s', '--pred_suffix', required=True, type=str, help='File suffix of the predictions.')

    args = parser.parse_args()
    evaluate_folder(args.ref, args.pred, args.ref_suffix, args.pred_suffix)


if __name__ == '__main__':
    main()
