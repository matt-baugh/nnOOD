import pickle
import json
from typing import Union, Any
from pathlib import Path

from nnood.paths import results_base, default_plans_identifier


def load_pickle(file_path: Union[str, Path], mode: str = 'rb'):
    with open(file_path, mode) as f:
        content = pickle.load(f)
    return content


def save_pickle(content: Any, file_path: Union[str, Path], mode: str = 'wb'):
    with open(file_path, mode) as f:
        pickle.dump(content, f)


def load_json(file_path: Union[str, Path]):
    with open(file_path) as f:
        content = json.load(f)
    return content


def save_json(content: dict, file_path: Union[str, Path], indent: int = 4, sort_keys: bool = True):
    with open(file_path, 'w') as f:
        json.dump(content, f, sort_keys=sort_keys, indent=indent)


def load_results_json(dataset: str, task: str, plans_identifier: str = default_plans_identifier):
    results_folder = Path(results_base, dataset, task, 'testResults', plans_identifier)
    return load_json(results_folder / 'summary.json')
