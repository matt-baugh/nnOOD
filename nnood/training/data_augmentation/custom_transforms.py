from batchgenerators.augmentations.crop_and_pad_augmentations import random_crop as random_crop_aug
from batchgenerators.augmentations.crop_and_pad_augmentations import center_crop as center_crop_aug
from batchgenerators.augmentations.utils import create_zero_centered_coordinate_mesh, elastic_deform_coordinates, \
    interpolate_img, \
    rotate_coords_2d, scale_coords, create_matrix_rotation_x_3d, create_matrix_rotation_y_3d, \
    create_matrix_rotation_z_3d
from batchgenerators.transforms.abstract_transforms import AbstractTransform
import numpy as np


def convert_3d_to_2d_generator(data_dict):
    shp = data_dict['data'].shape
    data_dict['data'] = data_dict['data'].reshape((shp[0], shp[1] * shp[2], shp[3], shp[4]))
    data_dict['coords'] = data_dict['coords'].reshape((shp[0], shp[1] * shp[2], shp[3], shp[4]))
    data_dict['orig_shape_data'] = shp
    return data_dict


def convert_2d_to_3d_generator(data_dict):
    shp = data_dict['orig_shape_data']
    current_shape = data_dict['data'].shape
    data_dict['data'] = data_dict['data'].reshape((shp[0], shp[1], shp[2], current_shape[-2], current_shape[-1]))
    data_dict['coords'] = data_dict['coords'].reshape((shp[0], shp[1], shp[2], current_shape[-2], current_shape[-1]))
    return data_dict


class Convert3DTo2DTransform(AbstractTransform):
    def __init__(self):
        pass

    def __call__(self, **data_dict):
        return convert_3d_to_2d_generator(data_dict)


class Convert2DTo3DTransform(AbstractTransform):
    def __init__(self):
        pass

    def __call__(self, **data_dict):
        return convert_2d_to_3d_generator(data_dict)


# Need to reimplement SpatialTransform, to allow the label NOT to be treated as segmentation
# To avoid bugs, classes are copied from batch generators BUT with flag added to disable seg behaviour
# variable names (inc. 'seg') are unchanged to make comparisons easier.
class MySpatialTransform(AbstractTransform):
    """The ultimate spatial transform generator. Rotation, deformation, scaling, cropping: It has all you ever dreamed
    of. Computational time scales only with patch_size, not with input patch size or type of augmentations used.
    Internally, this transform will use a coordinate grid of shape patch_size to which the transformations are
    applied (very fast). Interpolation on the image data will only be done at the very end

    Args:
        patch_size (tuple/list/ndarray of int): Output patch size

        label_is_seg (bool): whether label should be treated as segmentation.

        patch_center_dist_from_border (tuple/list/ndarray of int, or int): How far should the center pixel of the
        extracted patch be from the image border? Recommended to use patch_size//2.
        This only applies when random_crop=True

        do_elastic_deform (bool): Whether or not to apply elastic deformation

        alpha (tuple of float): magnitude of the elastic deformation; randomly sampled from interval

        sigma (tuple of float): scale of the elastic deformation (small = local, large = global); randomly sampled
        from interval

        do_rotation (bool): Whether or not to apply rotation

        angle_x, angle_y, angle_z (tuple of float): angle in rad; randomly sampled from interval. Always double check
        whether axes are correct!

        do_scale (bool): Whether or not to apply scaling

        scale (tuple of float): scale range ; scale is randomly sampled from interval

        border_mode_data: How to treat border pixels in data? see scipy.ndimage.map_coordinates

        border_cval_data: If border_mode_data=constant, what value to use?

        order_data: Order of interpolation for data. see scipy.ndimage.map_coordinates

        border_mode_seg: How to treat border pixels in seg? see scipy.ndimage.map_coordinates

        border_cval_seg: If border_mode_seg=constant, what value to use?

        order_seg: Order of interpolation for seg. see scipy.ndimage.map_coordinates. Strongly recommended to use 0!
        If !=0 then you will have to round to int and also beware of interpolation artifacts if you have more then
        labels 0 and 1. (for example if you have [0, 0, 0, 2, 2, 1, 0] the neighboring [0, 0, 2] bay result in
        [0, 1, 2])

        random_crop: True: do a random crop of size patch_size and minimal distance to border of
        patch_center_dist_from_border. False: do a center crop of size patch_size

        independent_scale_for_each_axis: If True, a scale factor will be chosen independently for each axis.
    """

    def __init__(self, patch_size, label_is_seg, patch_center_dist_from_border=30,
                 do_elastic_deform=True, alpha=(0., 1000.), sigma=(10., 13.),
                 do_rotation=True, angle_x=(0, 2 * np.pi), angle_y=(0, 2 * np.pi), angle_z=(0, 2 * np.pi),
                 do_scale=True, scale=(0.75, 1.25), border_mode_data='nearest', border_cval_data=0, order_data=3,
                 border_mode_seg='constant', border_cval_seg=0, order_seg=0, random_crop=True, data_key='data',
                 label_key='seg', pos_enc_key='coords', p_el_per_sample=1, p_scale_per_sample=1, p_rot_per_sample=1,
                 independent_scale_for_each_axis=False, p_rot_per_axis: float = 1,
                 p_independent_scale_per_axis: int = 1):

        self.label_is_seg = label_is_seg
        self.independent_scale_for_each_axis = independent_scale_for_each_axis
        self.p_rot_per_sample = p_rot_per_sample
        self.p_scale_per_sample = p_scale_per_sample
        self.p_el_per_sample = p_el_per_sample
        self.data_key = data_key
        self.label_key = label_key
        self.pos_enc_key = pos_enc_key
        self.patch_size = patch_size
        self.patch_center_dist_from_border = patch_center_dist_from_border
        self.do_elastic_deform = do_elastic_deform
        self.alpha = alpha
        self.sigma = sigma
        self.do_rotation = do_rotation
        self.angle_x = angle_x
        self.angle_y = angle_y
        self.angle_z = angle_z
        self.do_scale = do_scale
        self.scale = scale
        self.border_mode_data = border_mode_data
        self.border_cval_data = border_cval_data
        self.order_data = order_data
        self.border_mode_seg = border_mode_seg
        self.border_cval_seg = border_cval_seg
        self.order_seg = order_seg
        self.random_crop = random_crop
        self.p_rot_per_axis = p_rot_per_axis
        self.p_independent_scale_per_axis = p_independent_scale_per_axis

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)
        seg = data_dict.get(self.label_key)
        pos_enc = data_dict.get(self.pos_enc_key)

        if self.patch_size is None:
            if len(data.shape) == 4:
                patch_size = (data.shape[2], data.shape[3])
            elif len(data.shape) == 5:
                patch_size = (data.shape[2], data.shape[3], data.shape[4])
            else:
                raise ValueError('only support 2D/3D batch data.')
        else:
            patch_size = self.patch_size

        ret_val = augment_spatial(data, seg, pos_enc, patch_size=patch_size, seg_is_seg=self.label_is_seg,
                                  patch_center_dist_from_border=self.patch_center_dist_from_border,
                                  do_elastic_deform=self.do_elastic_deform, alpha=self.alpha, sigma=self.sigma,
                                  do_rotation=self.do_rotation, angle_x=self.angle_x, angle_y=self.angle_y,
                                  angle_z=self.angle_z, do_scale=self.do_scale, scale=self.scale,
                                  border_mode_data=self.border_mode_data,
                                  border_cval_data=self.border_cval_data, order_data=self.order_data,
                                  border_mode_seg=self.border_mode_seg, border_cval_seg=self.border_cval_seg,
                                  order_seg=self.order_seg, random_crop=self.random_crop,
                                  p_el_per_sample=self.p_el_per_sample, p_scale_per_sample=self.p_scale_per_sample,
                                  p_rot_per_sample=self.p_rot_per_sample,
                                  independent_scale_for_each_axis=self.independent_scale_for_each_axis,
                                  p_rot_per_axis=self.p_rot_per_axis,
                                  p_independent_scale_per_axis=self.p_independent_scale_per_axis)
        data_dict[self.data_key] = ret_val[0]
        data_dict[self.pos_enc_key] = ret_val[1]
        if seg is not None:
            data_dict[self.label_key] = ret_val[2]

        return data_dict


def augment_spatial(data: np.ndarray, seg: np.ndarray, pos_enc: np.ndarray, patch_size, seg_is_seg,
                    patch_center_dist_from_border=30, do_elastic_deform=True, alpha=(0., 1000.), sigma=(10., 13.),
                    do_rotation=True, angle_x=(0, 2 * np.pi), angle_y=(0, 2 * np.pi), angle_z=(0, 2 * np.pi),
                    do_scale=True, scale=(0.75, 1.25), border_mode_data='nearest', border_cval_data=0, order_data=3,
                    border_mode_seg='constant', border_cval_seg=0, order_seg=0, random_crop=True, p_el_per_sample=1,
                    p_scale_per_sample=1, p_rot_per_sample=1, independent_scale_for_each_axis=False,
                    p_rot_per_axis: float = 1, p_independent_scale_per_axis: int = 1):
    dim = len(patch_size)
    seg_result = None
    if seg is not None:
        if dim == 2:
            seg_result = np.zeros((seg.shape[0], seg.shape[1], patch_size[0], patch_size[1]), dtype=np.float32)
        else:
            seg_result = np.zeros((seg.shape[0], seg.shape[1], patch_size[0], patch_size[1], patch_size[2]),
                                  dtype=np.float32)

    if dim == 2:
        data_result = np.zeros((data.shape[0], data.shape[1], patch_size[0], patch_size[1]), dtype=np.float32)
    else:
        data_result = np.zeros((data.shape[0], data.shape[1], patch_size[0], patch_size[1], patch_size[2]),
                               dtype=np.float32)

    pos_enc_result = np.zeros((pos_enc.shape[0], pos_enc.shape[1], *patch_size), dtype=np.float32)

    if not isinstance(patch_center_dist_from_border, (list, tuple, np.ndarray)):
        patch_center_dist_from_border = dim * [patch_center_dist_from_border]

    for sample_id in range(data.shape[0]):
        coords = create_zero_centered_coordinate_mesh(patch_size)
        modified_coords = False

        # Get coordinate of centre of patch, as np.ndarray of ([z, ], y, x)
        patch_centre_coord: np.ndarray = pos_enc[(sample_id, slice(pos_enc.shape[1]),
                                                  *(np.array(pos_enc.shape)[2:] // 2))]
        orig_patch_centre_coord = patch_centre_coord.copy()

        # As elastic deformation gives random perturbation at pixel level, assume it does not change the patch position
        if do_elastic_deform and np.random.uniform() < p_el_per_sample:
            a = np.random.uniform(alpha[0], alpha[1])
            s = np.random.uniform(sigma[0], sigma[1])
            coords = elastic_deform_coordinates(coords, a, s)
            modified_coords = True

        if do_rotation and np.random.uniform() < p_rot_per_sample:

            if np.random.uniform() <= p_rot_per_axis:
                a_x = np.random.uniform(angle_x[0], angle_x[1])
            else:
                a_x = 0

            if dim == 3:
                if np.random.uniform() <= p_rot_per_axis:
                    a_y = np.random.uniform(angle_y[0], angle_y[1])
                else:
                    a_y = 0

                if np.random.uniform() <= p_rot_per_axis:
                    a_z = np.random.uniform(angle_z[0], angle_z[1])
                else:
                    a_z = 0

                rot_matrix = np.identity(len(coords))
                rot_matrix = create_matrix_rotation_x_3d(a_x, rot_matrix)
                rot_matrix = create_matrix_rotation_y_3d(a_y, rot_matrix)
                rot_matrix = create_matrix_rotation_z_3d(a_z, rot_matrix)
                coords = np.dot(coords.reshape(len(coords), -1).transpose(), rot_matrix).transpose().reshape(
                    coords.shape)
                # The coordinates are transformed to say where that point should be sampled FROM.
                # For the centre coord we want where this point is moved TOO, hence it's the opposite rotation (which is
                # the same as applying the inverse operation).
                patch_centre_coord = patch_centre_coord @ np.linalg.inv(rot_matrix)
            else:
                coords = rotate_coords_2d(coords, a_x)
                # Same as above, but as it's 2D we can directly get the opposite transformation, probably faster?
                patch_centre_coord = rotate_coords_2d(patch_centre_coord, -a_x)
            modified_coords = True

        if do_scale and np.random.uniform() < p_scale_per_sample:
            if independent_scale_for_each_axis and np.random.uniform() < p_independent_scale_per_axis:
                sc = []
                for _ in range(dim):
                    if np.random.random() < 0.5 and scale[0] < 1:
                        sc.append(np.random.uniform(scale[0], 1))
                    else:
                        sc.append(np.random.uniform(max(scale[0], 1), scale[1]))
            else:
                if np.random.random() < 0.5 and scale[0] < 1:
                    sc = np.random.uniform(scale[0], 1)
                else:
                    sc = np.random.uniform(max(scale[0], 1), scale[1])

            coords = scale_coords(coords, sc)

            # Again, we are tracking where the centre moves TO, so we do opposite of scale_coords.
            patch_centre_coord /= sc
            modified_coords = True

        patch_movement = patch_centre_coord - orig_patch_centre_coord
        reshaped_patch_movement = patch_movement.reshape((len(patch_movement), *([1] * len(patch_centre_coord))))
        transformed_pos_enc = pos_enc[sample_id] + reshaped_patch_movement

        # Only allow centre cropping, people should never random crop as already taking random patches, so no reward,
        # plus difficult to cover all below cases.
        # First [0] is to get 'data' from returned tuple, second is to take out of fake batch
        pos_enc_result[sample_id] = center_crop_aug(transformed_pos_enc[None], patch_size)[0][0]

        # now find a nice center location
        if modified_coords:
            for d in range(dim):
                if random_crop:
                    ctr = np.random.uniform(patch_center_dist_from_border[d],
                                            data.shape[d + 2] - patch_center_dist_from_border[d])
                    assert False, 'Random crop augmentation not implemented for positional encoding, we\'re already ' \
                                  'using random patches, what purpose does random cropping serve??'
                else:
                    ctr = data.shape[d + 2] / 2. - 0.5
                coords[d] += ctr
            for channel_id in range(data.shape[1]):
                data_result[sample_id, channel_id] = interpolate_img(data[sample_id, channel_id], coords, order_data,
                                                                     border_mode_data, cval=border_cval_data)
            if seg is not None:
                for channel_id in range(seg.shape[1]):
                    seg_result[sample_id, channel_id] = interpolate_img(seg[sample_id, channel_id], coords, order_seg,
                                                                        border_mode_seg, cval=border_cval_seg,
                                                                        is_seg=seg_is_seg)
        else:
            if seg is None:
                s = None
            else:
                s = seg[sample_id:sample_id + 1]
            if random_crop:
                margin = [patch_center_dist_from_border[d] - patch_size[d] // 2 for d in range(dim)]
                d, s = random_crop_aug(data[sample_id:sample_id + 1], s, patch_size, margin)
                assert False, 'Random crop augmentation not implemented for positional encoding, we\'re already ' \
                              'using random patches, what purpose does random cropping serve??'
            else:
                d, s = center_crop_aug(data[sample_id:sample_id + 1], patch_size, s)
            data_result[sample_id] = d[0]
            if seg is not None:
                seg_result[sample_id] = s[0]
    return data_result, pos_enc_result, seg_result


# Need to reimplement batchgenerator's MirrorTransform, to incorporate and update the positional encoding.
# To avoid bugs, code is edited as little as possible, including maintaining variable names like 'seg', to make
# comparisons easier.
class MyMirrorTransform(AbstractTransform):
    """ Randomly mirrors data along specified axes. Mirroring is evenly distributed. Probability of mirroring along
    each axis is 0.5

    Args:
        axes (tuple of int): axes along which to mirror

    """

    def __init__(self, axes=(0, 1, 2), data_key='data', label_key='seg', pos_enc_key='coords', p_per_sample=1):
        self.p_per_sample = p_per_sample
        self.data_key = data_key
        self.label_key = label_key
        self.pos_enc_key = pos_enc_key
        self.axes = axes
        if max(axes) > 2:
            raise ValueError('MirrorTransform now takes the axes as the spatial dimensions. What previously was '
                             'axes=(2, 3, 4) to mirror along all spatial dimensions of a 5d tensor (b, c, x, y, z) '
                             'is now axes=(0, 1, 2). Please adapt your scripts accordingly.')

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)
        coords = data_dict.get(self.pos_enc_key)
        seg = data_dict.get(self.label_key)

        for b in range(len(data)):
            if np.random.uniform() < self.p_per_sample:
                sample_seg = None
                if seg is not None:
                    sample_seg = seg[b]
                ret_val = augment_mirroring(data[b], coords[b], sample_seg, axes=self.axes)
                data[b] = ret_val[0]
                coords[b] = ret_val[1]
                if seg is not None:
                    seg[b] = ret_val[2]

        data_dict[self.data_key] = data
        if seg is not None:
            data_dict[self.label_key] = seg

        return data_dict


def augment_mirroring(sample_data, sample_coords, sample_seg=None, axes=(0, 1, 2)):
    if (len(sample_data.shape) != 3) and (len(sample_data.shape) != 4):
        raise Exception(
            'Invalid dimension for sample_data and sample_seg. sample_data and sample_seg should be either '
            '[channels, x, y] or [channels, x, y, z]')
    if 0 in axes and np.random.uniform() < 0.5:
        sample_data[:, :] = sample_data[:, ::-1]
        # As coordinates range from -1 to 1, mirroring is the equivalent of inverting them. This represents where the
        # patch would appear if the mirrored patch would appear in the original sample.
        sample_coords[0] = -sample_coords[0]
        if sample_seg is not None:
            sample_seg[:, :] = sample_seg[:, ::-1]
    if 1 in axes and np.random.uniform() < 0.5:
        sample_data[:, :, :] = sample_data[:, :, ::-1]
        sample_coords[1] = -sample_coords[1]
        if sample_seg is not None:
            sample_seg[:, :, :] = sample_seg[:, :, ::-1]
    if 2 in axes and len(sample_data.shape) == 4:
        if np.random.uniform() < 0.5:
            sample_data[:, :, :, :] = sample_data[:, :, :, ::-1]
            sample_coords[2] = -sample_coords[2]
            if sample_seg is not None:
                sample_seg[:, :, :, :] = sample_seg[:, :, :, ::-1]
    return sample_data, sample_coords, sample_seg
