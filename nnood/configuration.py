import os

THREAD_NUM_VAR = 'nnood_def_n_proc'

default_num_processes = int(os.environ[THREAD_NUM_VAR]) if THREAD_NUM_VAR in os.environ else 8
RESAMPLING_SEPARATE_Z_ANISO_THRESHOLD = 3
