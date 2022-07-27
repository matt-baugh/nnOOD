from setuptools import setup

setup(
    name='nnood',
    version='',
    packages=['nnood', 'nnood.data', 'nnood.data.dataset_conversion', 'nnood.utils', 'nnood.training',
              'nnood.training.dataloading', 'nnood.training.loss_functions', 'nnood.training.network_training',
              'nnood.training.data_augmentation', 'nnood.inference', 'nnood.evaluation', 'nnood.preprocessing',
              'nnood.experiment_planning', 'nnood.network_architecture', 'nnood.self_supervised_task',
              'nnood.self_supervised_task.patch_transforms', 'tests', 'tests.self_supervised_task'],
    url='',
    license='',
    author='',
    author_email='',
    description='',
    install_requires=[
        'numpy',
        'nibabel',
        'SimpleITK',
        'tqdm',
        'opencv-python',
        'pandas',
        'torch>=1.10.0',
        'matplotlib',
        'sklearn',
        'scikit-learn>=1.0.1',
        'batchgenerators>=0.23',
        'scikit-image>=0.19.0',
        'argparse',
        'scipy',
        'unittest2'
    ]
)
