import argparse
import shutil
from pathlib import Path

from nnood.configuration import default_num_processes
from nnood.data.sanity_checks import verify_dataset_integrity
from nnood.paths import raw_data_base, preprocessed_data_base
from nnood.experiment_planning.DatasetAnalyser import DatasetAnalyser
from nnood.experiment_planning.experiment_planner import ExperimentPlanner

# Plan experiment, and convert dataset to .npz format (gathering modalities of each sample)
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_names', nargs='+', help='List of datasets you wish to run planning and'
                                                                 ' preprocessing for.')
    parser.add_argument('-no_pp', action='store_true',
                        help='Set this flag if you dont want to run the preprocessing. If this is set then this script '
                             'will only run the experiment planning and create the plans file')
    # parser.add_argument('-tl', type=int, required=False, default=8,
    #                     help='Number of processes used for preprocessing the low resolution data for the 3D low '
    #                          'resolution U-Net. This can be larger than -tf. Don't overdo it or you will run out of '
    #                          'RAM')
    parser.add_argument('-p', type=int, required=False, default=default_num_processes,
                        help='Number of processes used for preprocessing the full resolution data. Don\'t overdo it or '
                             'you will run out of RAM')
    parser.add_argument('--verify_dataset_integrity', required=False, default=False, action='store_true',
                        help='Set this flag to check the dataset integrity. This is useful and should be done once for '
                             'each dataset!')
    parser.add_argument('--disable_skip', type=int, required=False, default=0,
                        help='Number of skip connections to disable (starting from top of U-Net). Remember to change'
                             'plans identifier when generating new plans!')

    args = parser.parse_args()
    dataset_names = args.dataset_names
    run_preprocessing = not args.no_pp
    num_processes = args.p
    verify_dataset_integ = args.verify_dataset_integrity
    disable_skip = args.disable_skip

    raw_data_path = Path(raw_data_base)

    dataset_paths = []
    for d_n in dataset_names:
        d_path = raw_data_path / d_n

        if verify_dataset_integ:
            verify_dataset_integrity(d_path)

        dataset_paths.append(d_path)

    for d_path in dataset_paths:
        print('Planning for dataset at: ', d_path)

        print('Analysing dataset...')
        dataset_analyser = DatasetAnalyser(d_path, num_processes=num_processes)
        _ = dataset_analyser.analyse_dataset()

        print('Copying dataset.json and properties files')
        preprocessed_data_dir = Path(preprocessed_data_base) / d_path.name
        preprocessed_data_dir.mkdir(exist_ok=True)
        shutil.copy(d_path / 'dataset.json', preprocessed_data_dir)
        shutil.copy(d_path / 'dataset_properties.pkl', preprocessed_data_dir)

        print('Planning experiment...')
        exp_planner = ExperimentPlanner(d_path, preprocessed_data_dir, num_processes, disable_skip)
        exp_planner.plan_experiment()
        if run_preprocessing:
            print('Running preprocessing...')
            exp_planner.run_preprocessing()

    print('Done')
