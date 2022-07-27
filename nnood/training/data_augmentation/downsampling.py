from batchgenerators.transforms.abstract_transforms import AbstractTransform
import numpy as np
from skimage.transform import resize


class DownsampleSegForDSTransform(AbstractTransform):
    """
    data_dict['output_key'] will be a list of segmentations scaled according to ds_scales
    """
    def __init__(self, ds_scales=(1, 0.5, 0.25), order=1, input_key='seg', output_key='seg', axes=None):
        self.axes = axes
        self.output_key = output_key
        self.input_key = input_key
        self.order = order
        self.ds_scales = ds_scales

    def __call__(self, **data_dict):
        data_dict[self.output_key] = downsample_seg_for_ds_transform(data_dict[self.input_key], self.ds_scales,
                                                                     self.order, self.axes)
        return data_dict


def downsample_seg_for_ds_transform(seg, ds_scales=((1, 1, 1), (0.5, 0.5, 0.5), (0.25, 0.25, 0.25)), order=1,
                                    axes=None):
    if axes is None:
        axes = list(range(2, len(seg.shape)))

    output = []

    for s in ds_scales:

        if all([i == 1 for i in s]):
            output.append(seg)
        else:
            new_shape = np.array(seg.shape).astype(float)
            for i, a in enumerate(axes):
                new_shape[a] *= s[i]
            new_shape = np.round(new_shape).astype(int)
            out_seg = np.zeros(new_shape, dtype=seg.dtype)
            for b in range(seg.shape[0]):
                for c in range(seg.shape[1]):
                    out_seg[b, c] = resize(seg[b, c], new_shape[2:], order)
            output.append(out_seg)

    return output
