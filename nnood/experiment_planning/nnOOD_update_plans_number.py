import argparse
from pathlib import Path
import re
import shutil

from nnood.paths import default_plans_identifier, preprocessed_data_base

# Copy old plan to current plans identifier.
# Use if you've made a change in another aspect of the pipeline (like the trainer)
if __name__ == '__main__':

    match = re.search(r'(.*\.)(\d+)$', default_plans_identifier)
    assert match is not None, f"Default plans identifier doesn't match expected pattern: {default_plans_identifier}"
    plans_id_stem = match.group(1)
    curr_id_num = int(match.group(2))

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--copy_from', required=False, default=plans_id_stem + str(curr_id_num - 1),
                        help='Identifier of plans to copy from. Defaults to previous.')
    parser.add_argument('-d', '--dataset_names', nargs='+', help='List of datasets you wish to update the plans for.')

    args = parser.parse_args()
    copy_from = args.copy_from
    dataset_names = args.dataset_names

    print('Copying plans from: ', copy_from)
    print('New plans identifier: ', default_plans_identifier)
    print('Updating plans of datasets:')

    for d_n in dataset_names:
        print(d_n)

    preprocessed_data_path = Path(preprocessed_data_base)

    for d_n in dataset_names:
        preprocessed_data_dir = preprocessed_data_path / d_n
        assert preprocessed_data_dir.is_dir(), f'Missing directory for preprocessed data: {preprocessed_data_dir}'

        old_plans_path = preprocessed_data_dir / copy_from
        assert old_plans_path.is_file(), f'Missing plans to copy from: {old_plans_path}'

        new_plans_path = preprocessed_data_dir / default_plans_identifier
        assert not new_plans_path.is_file(), f'Plans with current identifier already exist: {new_plans_path}'

        shutil.copy(old_plans_path, new_plans_path)
    print('Done')
