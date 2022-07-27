import numpy as np

IMAGENET_STATS = {
    'png-r': (0.485, 0.456),
    'png-g': (0.456, 0.224),
    'png-b': (0.406, 0.225)
}

GLOBAL_NORMALISATION_MODALITIES = ['ct', 'png-bw']


def normalise_png_channel(channel_data: np.ndarray, channel: str, norm_fn):
    c_mean, c_std = IMAGENET_STATS[channel]
    return norm_fn(channel_data, c_mean, c_std)


# Either normalises or denormalises, depending on is_norm
def _norm_helper(data, normalisation_scheme_per_modality, intensity_properties, channel_properties, is_norm: bool):

    norm_fn = (lambda d, mean, std: (d - mean) / std) if is_norm else (lambda d, mean, std: d * std + mean)
    norm_fn_stable = lambda d, mean, std: norm_fn(d, mean, std + 1e-8)

    assert len(normalisation_scheme_per_modality) == len(data), 'self.normalisation_scheme_per_modality ' \
                                                                'must have as many entries as data has ' \
                                                                'modalities'

    result = []
    # Without a GT segmentation, we cannot use the same nonzero_mask for normalisation
    for c in range(len(data)):
        scheme = normalisation_scheme_per_modality[c]
        curr_channel = data[c]

        if scheme == 'ct':
            # clip to lb and ub from train data foreground and use foreground mn and sd from training data
            assert intensity_properties is not None, 'Cannot normalise CT without intensity properties'
            lower_bound = intensity_properties[c]['percentile_00_5']
            upper_bound = intensity_properties[c]['percentile_99_5']

            if is_norm:
                curr_channel = np.clip(curr_channel, lower_bound, upper_bound)
            else:
                print('WARNING: when denormalising a CT image we cannot invert the clipping, so result will not be '
                      'exact')

            result.append(norm_fn(curr_channel, intensity_properties[c]['mean'], intensity_properties[c]['sd']))
        elif scheme == 'global-z':
            assert intensity_properties is not None, f'Cannot normalise modality {c} without intensity properties'

            result.append(norm_fn(curr_channel, intensity_properties[c]['mean'], intensity_properties[c]['sd']))
        elif scheme == 'noNorm':
            pass
        elif scheme in IMAGENET_STATS.keys():
            result.append(normalise_png_channel(curr_channel, scheme, norm_fn))
        elif scheme == 'z-score':
            result.append(norm_fn_stable(curr_channel, channel_properties[c]['mean'], channel_properties[c]['sd']))
        else:
            assert False, f'Unrecognised normalisation scheme: {scheme}'

    return np.stack(result)


def normalise(data, normalisation_scheme_per_modality, intensity_properties, channel_properties):
    return _norm_helper(data, normalisation_scheme_per_modality, intensity_properties, channel_properties, True)


def denormalise(data, normalisation_scheme_per_modality, intensity_properties, channel_properties):
    return _norm_helper(data, normalisation_scheme_per_modality, intensity_properties, channel_properties, False)


def modality_norm_scheme(mod: str):

    if mod == 'CT' or mod == 'ct':
        return 'ct'
    elif mod in GLOBAL_NORMALISATION_MODALITIES:
        return 'global-z'
    elif mod == 'noNorm':
        return 'noNorm'
    elif mod in IMAGENET_STATS.keys():
        return mod
    else:
        return 'z-score'
