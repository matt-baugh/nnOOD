from collections import OrderedDict

# Some 'modalities' actually contain multiple modalities
MODALITY_TRANSLATOR = {
    'png': ('png-r', 'png-g', 'png-b'),
    'ct': ('ct',),
    'mri': ('mri',),
    'png-bw': ('png-bw',),  # Greyscale image
    'png-xray': ('png-xray',)
}


def num_modality_components(mod: str):

    if mod not in MODALITY_TRANSLATOR:
        print(f'Unknown modality {mod}, assuming it only has 1 component')
        return 1

    return len(MODALITY_TRANSLATOR[mod])


def get_channel_list(modalities: OrderedDict):
    channel_list = []
    for i in modalities.keys():
        mod_components = MODALITY_TRANSLATOR[modalities[i]]
        for m in mod_components:
            channel_list.append(m)
    return channel_list
