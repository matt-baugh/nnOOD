import os
from typing import Optional
from pathlib import Path

RAW_ENVIRON_VAR = 'nnood_raw_data_base'
PREPROCESSED_ENVIRON_VAR = 'nnood_preprocessed_data_base'
RESULTS_ENVIRON_VAR = 'nnood_results_base'

DATASET_JSON_FILE = 'dataset.json'
DATASET_PROPERTIES_FILE = 'dataset_properties.pkl'

default_plans_identifier = 'nnood_plans_v1.0'
default_data_identifier = 'nnood_data_v1.0'

def setup_directory(var_name: str) -> Optional[str]:
    var_value = os.environ[var_name] if var_name in os.environ else None

    if var_value is not None:
        dir_path = Path(var_value)
        dir_path.mkdir(parents=True, exist_ok=True)
    else:
        print(var_name + ' is not defined, preventing nnood from completing any actions involving it\'s files')
    return var_value


raw_data_base = setup_directory(RAW_ENVIRON_VAR)
preprocessed_data_base = setup_directory(PREPROCESSED_ENVIRON_VAR)
results_base = setup_directory(RESULTS_ENVIRON_VAR)
