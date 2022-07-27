# Contains code for dataset conversion

I recommend using dataset_conversion/utils.generate_dataset_json to match correct format.

dataset.json components: 
 - name: str - dataset name
 - description: str - dataset description
 - reference: str - reference for dataset source
 - licence: str - dataset licence
 - release: str - date of dataset release
 - tensorImageSize: str - dimensionality of data including channels, either '3D' or '4D'
 - tensorImageSize: Dict[str, str] - dictionary matching file number to modality
 - numTraining: int - number of training examples
 - numTest: int - number of test examples
 - training: List[str] - list of training sample ids
 - test: List[str] - list of test sample ids
 - data_augs: Dict[str, Dict[str, Any]] - Dictionary, mapping data augmentation name to parameters and values.
   - Possible transforms + parameters:
      - elastic:
        - deform_alpha: List[int] - Alpha values range e.g. [0,900]
        - deform_sigma: List[int] - Sigma values range e.g. [9, 13]
      - scaling:
        - scale_range: List[float] - Scale factor range e.g. [0.85, 1.25]
      - rotation:
        - rot_max: int - Maximum amount of rotation around axis e.g. 15
      - gamma:
        - gamma_range: List[float] - Range of gamma transform values e.g. [0.7, 1.5]
      - mirror:
        - mirror_axes: List[int] - List of valid axes to mirror, where axis 0 is the axis immediately after the channels 
        dimension (y axis for 2D images, z axis for 3D) e.g. [0, 1, 2].
      - additive_brightness:
        - additive_brightness_mu: float - Mean of additive brightness nouse
        - additive_brightness_sigma: float - Standard deviation of additive brightness nouse
      - Potentially implemented in future:
        - gaussian_noise
        - gaussian_blur
        - brightness_multiplicative
        - contrast_aug
        - sim_low_res
