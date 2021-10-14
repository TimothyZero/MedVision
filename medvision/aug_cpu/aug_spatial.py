#  Copyright (c) 2020. The Medical Image Computing (MIC) Lab, 陶豪毅
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from typing import Union, List, Tuple
import gc
from copy import deepcopy
import cv2 as cv
import warnings
import itertools
import numpy as np
from numpy.random import randint
import scipy.ndimage as ndi
import random
import SimpleITK as sitk
from skimage.transform import resize

from .utils import cropBBoxes, clipBBoxes, objs2bboxes, bboxes2objs
from .base import AugBase


class Resize(AugBase):
    """
    support segmentation, detection and classification tasks
    support 2D and 3D image
    support forward and backward
    """

    def __init__(self, spacing=None, scale=None, factor=None):
        super().__init__()
        self.always = True
        self.spacing = spacing
        self.scale = scale
        self.factor = factor
        assert sum([spacing is not None, scale is not None, factor is not None]) == 1

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(spacing={}, scale={}, factor={})'.format(self.spacing, self.scale, self.factor)
        return repr_str

    @property
    def canBackward(self):
        return True

    def _forward_params(self, result):
        self._init_params(result)
        if self.spacing is not None:
            assert len(self.spacing) == self.dim, f"not match with a {result['img_dim']}D image"
            img_spacing = np.array(result['img_spacing'])
            new_spacing = np.array(self.spacing)
            new_spacing = np.where(new_spacing == -1, img_spacing, new_spacing)
            scale_factor = tuple(img_spacing / new_spacing)
        elif self.scale is not None:
            assert len(self.scale) == self.dim, f"not match with a {result['img_dim']}D image"
            new_shape = np.array(self.scale)
            new_shape = np.where(new_shape == -1, self.image_shape, new_shape)
            scale_factor = tuple(new_shape / self.image_shape)
        else:
            assert len(self.factor) == self.dim, f"not match with a {result['img_dim']}D image"
            scale_factor = tuple(self.factor)

        # 2d (0.5, 0.5)
        # 3d (0.5, 0.5, 0.5)
        self.params = scale_factor
        result[self.key_name] = self.params

    def _backward_params(self, result):
        self._init_params(result)
        params = result.pop(self.key_name, None)
        if params:
            # 2d (2.0, 2.0)
            # 3d (2.0, 2.0, 2.0)
            self.params = tuple(1 / np.array(params))

    def apply_to_img(self, result):
        for key in result.get('img_fields', []):
            img = result[key]  # .astype(np.float32)
            if not all([i == 1.0 for i in self.params]):
                # img = ndi.zoom(img, (1,) + self.params, order=2)
                new_shape = (self.channels,) + tuple(np.round(self.image_shape * np.array(self.params)).astype(np.int))
                if self.dim == 2:
                    img = cv.resize(np.swapaxes(img, 0, -1), new_shape[1:])
                    img = np.swapaxes(img, -1, 0)
                    if img.ndim == 2:
                        img = img[None, ...]
                else:
                    # img = resize(img, new_shape, order=2)  # faster than zoom but may cause wrong behaviour
                    img = ndi.zoom(img, (1,) + self.params, order=2)
            result[key] = img  # .astype(np.float32)
            if key == 'img':
                result['img_shape'] = img.shape
                result['img_spacing'] = tuple(np.array(result['img_spacing']) / np.array(self.params))

    def apply_to_seg(self, result):
        for key in result.get('seg_fields', []):
            if not all([i == 1.0 for i in self.params]):
                result[key] = ndi.zoom(result[key], (1,) + self.params, order=0)

                """ WARNING: pay attention to data type, different type will cause different behaviour """
                # scale_shape = np.round(np.array(result[key].shape) * np.array(self.params + (1.0,)))
                # result[key] = resize(result[key], tuple(scale_shape), mode="edge", order=0, preserve_range=True)
                # result[key] = rescale(result[key], self.params, mode="edge", order=0, preserve_range=True)

    def apply_to_det(self, result):
        for key in result.get('det_fields', []):
            if not all([i == 1.0 for i in self.params]):
                result[key][:, :2 * self.dim] = result[key][:, :2 * self.dim] * np.hstack(self.params[::-1] * 2)


class Pad(AugBase):
    """
    support segmentation, detection and classification tasks
    support 2D and 3D images
    support forward and backward
    """

    def __init__(self, size, mode='constant', val=0, center=True):
        super().__init__()
        self.always = True
        self.size = size
        self.mode = mode
        self.val = val
        self.center = center

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(size={}, mode={}, val={}, center=())'.format(self.size, self.mode, self.val, self.center)
        return repr_str

    @property
    def canBackward(self):
        return True

    def _forward_params(self, result):
        self._init_params(result)
        assert len(self.size) == self.dim, "image_scale must has same rank as image_shape"
        diff = np.maximum(np.array(self.size) - np.array(self.image_shape), 0)
        if self.center:
            diff = tuple(zip(np.array(diff) // 2, np.array(diff) - np.array(diff) // 2))
        else:
            diff = tuple(zip(np.zeros_like(diff), np.array(diff)))

        # 2d [(32, 32), (32, 32)]
        # 3d [(32, 32), (32, 32), (32, 32)]
        self.params = diff
        result[self.key_name] = self.params

    def _backward_params(self, result):
        self._init_params(result)
        params = result.pop(self.key_name, None)
        if params:
            # 2d [(32, 32), (32, 32)]
            # 3d [(32, 32), (32, 32), (32, 32)]
            self.params = params

    def apply_to_img(self, result):
        for key in result.get('img_fields', []):
            img = result[key]
            if self.isForwarding:
                if not all([i == 0 for i in self.params]):
                    img = np.pad(img, ((0, 0),) + self.params, mode=self.mode, constant_values=self.val)
            else:
                slices = tuple([slice(lp, shape - rp) for (lp, rp), shape in zip(self.params, self.image_shape)])
                slices = (slice(None),) + slices
                img = img[slices]

            result[key] = img
            if key == 'img':
                result['img_shape'] = img.shape

    def apply_to_seg(self, result):
        for key in result.get('seg_fields', []):
            if self.isForwarding:
                result[key] = np.pad(result[key], ((0, 0),) + self.params, mode=self.mode)
            else:
                slices = tuple([slice(lp, shape - rp) for (lp, rp), shape in zip(self.params, self.image_shape)])
                slices = (slice(None),) + slices
                result[key] = result[key][slices]

    def apply_to_det(self, result):
        # 2d [(32, 42), (33, 43)]
        # forward  [33, 32]
        # backward [-33, -32]
        ops = 1 if self.isForwarding else -1
        offsets = [i[0] * ops for i in self.params[::-1]] * 2
        for key in result.get('det_fields', []):
            result[key][:, :2 * self.dim] = result[key][:, :2 * self.dim] + np.array(offsets)


class RandomFlip(AugBase):
    """
    support segmentation, detection and classification tasks
    support 2D and 3D images
    support forward and backward
    """

    def __init__(self, p, axes=None):
        super().__init__()
        self.p = p
        self.axes = axes
        self.flip_p = 0.5

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(p={})'.format(self.p)
        return repr_str

    @property
    def canBackward(self):
        return True

    def _forward_params(self, result):
        self._init_params(result)
        # 2d [-1, 1]
        # 3d [1, -1, 1] , -1 = flip
        self.params = random.choices([-1, 1], weights=[self.flip_p, 1 - self.flip_p], k=self.dim)
        result[self.key_name] = tuple(self.params)

    def _backward_params(self, result):
        self._init_params(result)
        params = result.pop(self.key_name, None)
        if params:
            # 2d [-1, 1]
            # 3d [1, -1, 1]
            self.params = params

    def apply_to_img(self, result):
        slices = tuple(map(slice, [None] * self.dim, [None] * self.dim, self.params))
        slices = (slice(None),) + slices
        for key in result.get('img_fields', []):
            result[key] = result[key][slices]

    def apply_to_seg(self, result):
        slices = tuple(map(slice, [None] * self.dim, [None] * self.dim, self.params))
        slices = (slice(None),) + slices
        for key in result.get('seg_fields', []):
            result[key] = result[key][slices]

    def apply_to_det(self, result):
        for key in result.get('det_fields', []):
            bboxes = np.array(result[key])
            for i, tag in enumerate(self.params[::-1]):  # xyz
                if tag == -1:
                    bboxes[:, i] = self.image_shape[- i - 1] - bboxes[:, i] - 1
                    bboxes[:, i + self.dim] = self.image_shape[- i - 1] - bboxes[:, i + self.dim] - 1
                    bboxes[:, [i, i + self.dim]] = bboxes[:, [i + self.dim, i]]
            result[key] = bboxes


class RandomScale(AugBase):
    """
    support segmentation, detection and classification tasks
    support 2D and 3D image
    support forward and backward
    """

    def __init__(self, p, factor):
        super().__init__()
        self.p = p
        self.factor = factor
        self._tmp_params = None

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(factor={})'.format(self.factor)
        return repr_str

    @property
    def canBackward(self):
        return True

    def _forward_params(self, result):
        self._init_params(result)
        scales = (self.get_range(self.factor, 1),) * self.dim
        # 2d (0.5, 0.5)
        # 3d (0.5, 0.5, 0.5)
        self.params = scales
        result[self.key_name] = self.params

    def _backward_params(self, result):
        self._init_params(result)
        params = result.pop(self.key_name, None)
        if params:
            # 2d (2.0, 2.0)
            # 3d (2.0, 2.0, 2.0)
            self.params = tuple(1 / np.array(params))

    def apply_to_img(self, result):
        for key in result.get('img_fields', []):
            img = result[key]
            p = np.zeros_like(img)
            if not all([i == 1.0 for i in self.params]):
                new_shape = (self.channels,) + tuple(np.round(self.image_shape * np.array(self.params)))
                img = resize(img, tuple(new_shape), order=1)
                # print(img.shape, p.shape)
                if self.params[0] <= 1.0:
                    diff = np.array(p.shape[1:]) - np.array(img.shape[1:])
                    start, end = diff // 2, diff // 2 - diff
                    start = [None if i == 0 else i for i in start]  # while diff = 0,0
                    end = [None if i == 0 else i for i in end]  # while diff = 0,0
                    slices = (slice(None),) + tuple(map(slice, start, end))
                    assert p[
                               slices].shape == img.shape, f'{p[slices].shape}-{img.shape}-{p.shape}, diff={diff}, {self.params}'
                    p[slices] = img
                    self._tmp_params = (slices, True)
                else:
                    diff = np.array(img.shape[1:]) - np.array(p.shape[1:])
                    start, end = diff // 2, diff // 2 - diff
                    start = [None if i == 0 else i for i in start]  # while diff = 0,0
                    end = [None if i == 0 else i for i in end]  # while diff = 0,0
                    slices = (slice(None),) + tuple(map(slice, start, end))
                    assert img[
                               slices].shape == p.shape, f'{img[slices].shape}-{img.shape}-{p.shape}, diff={diff}, {self.params}'
                    p = img[slices]
                    self._tmp_params = (slices, False)
                # print(diff, slices)
                result[key] = p
                if key == 'img':
                    result['img_shape'] = p.shape
                    result['img_spacing'] = tuple(np.array(result['img_spacing']) / np.array(self.params))

    def apply_to_seg(self, result):
        slices, flag = self._tmp_params
        for key in result.get('seg_fields', []):
            p = np.zeros_like(result[key])
            if not all([i == 1.0 for i in self.params]):
                img = ndi.zoom(result[key], (1,) + self.params, order=0)
                if flag:
                    p[slices] = img
                else:
                    p = img[slices]
                result[key] = p

                """ WARNING: pay attention to data type, different type will cause different behaviour """
                # scale_shape = np.round(np.array(result[key].shape) * np.array(result['scale_factor'] + (1.0,)))
                # result[key] = resize(result[key], tuple(scale_shape), mode="edge", order=0, preserve_range=True)
                # result[key] = rescale(result[key], result['scale_factor'], mode="edge", order=0, preserve_range=True)

    def apply_to_det(self, result):
        slices, flag = self._tmp_params
        slices = slices[1:]
        for key in result.get('det_fields', []):
            if not all([i == 1.0 for i in self.params]):
                # print("zoom", key)
                ops = -1 if flag else 1
                offsets = np.array([ops * i.start if i.start is not None else 0 for i in slices[::-1]] * 2)
                result[key][:, :2 * self.dim] = result[key][:, :2 * self.dim] * np.hstack(
                    self.params[::-1] * 2) - offsets
                result[key] = clipBBoxes(self.dim, result[key], self.image_shape)


class RandomShift(AugBase):
    """
    support segmentation, detection and classification tasks
    support 2D and 3D images
    support forward and backward
    """

    def __init__(self, p, shift=0.1, mode='constant'):
        super().__init__()
        self.p = p
        self.shift = shift
        self.mode = mode

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(p={}, shift={}, mode={})'.format(self.p, self.shift, self.mode)
        return repr_str

    @property
    def canBackward(self):
        return True

    def _forward_params(self, result):
        self._init_params(result)
        if self.dim == 3:  # only shift on axes (xy)
            random_shift_ratio = [0] + list([self.get_range(self.shift)] * (self.dim - 1))
            shift = (0,) + tuple(np.round(np.array(self.image_shape) * np.array(random_shift_ratio)))
            shift = tuple(shift)
        else:
            random_shift_ratio = list([self.get_range(self.shift)] * self.dim)
            shift = (0,) + tuple(np.round(np.array(self.image_shape) * np.array(random_shift_ratio)))
            shift = tuple(shift)

        # 2d (14, 35)
        # 3d (0, 14, 35)
        self.params = shift
        result[self.key_name] = self.params

    def _backward_params(self, result):
        self._init_params(result)
        params = result.pop(self.key_name, None)
        if params:
            # 2d (-14, -35)
            # 3d (0, -14, -35)
            self.params = tuple(- np.array(params))

    def apply_to_img(self, result):
        for key in result.get('img_fields', []):
            result[key] = ndi.shift(result[key], self.params, order=1, mode=self.mode)

    def apply_to_seg(self, result):
        for key in result.get('seg_fields', []):
            result[key] = ndi.shift(result[key], self.params, order=0, mode=self.mode)

    def apply_to_det(self, result):
        offset = self.params[1:] * 2
        for key in result.get('det_fields', []):
            result[key][:, :2 * self.dim] = result[key][:, :2 * self.dim] + offset[::-1]
            result[key] = clipBBoxes(self.dim, result[key], self.image_shape)


class RandomRotate(AugBase):
    """
    support segmentation, detection and classification tasks
    support 2D and 3D images
    support forward and backward
    """

    def __init__(self, p, angle, axes=None, reshape=False, order=1, mode='constant', val=0.0):
        super().__init__()
        self.p = p
        self.angle = angle
        self.axes = axes
        self.reshape = reshape
        self.order = order
        self.mode = mode
        self.val = val

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(p={}, angle={}, axes={}, order={}, mode={})'.format(
            self.p, self.angle, self.axes, self.order, self.mode)
        return repr_str

    @property
    def canBackward(self):
        return True

    def _forward_params(self, result):
        self._init_params(result)
        if self.axes is None:
            self.axes = result['img_dim'] == 3 and (3, 2) or (2, 1)
        else:
            self.axes = sorted(self.axes, reverse=True)  # x,y,z order
        # 40 deg
        self.params = self.get_range(self.angle)
        result[self.key_name] = self.params

    def _backward_params(self, result):
        self._init_params(result)
        params = result.pop(self.key_name, None)
        if params:
            if self.axes is None:
                self.axes = result['img_dim'] == 3 and (3, 2) or (2, 1)
            else:
                self.axes = sorted(self.axes, reverse=True)  # x,y,z order
            # -40 deg
            self.params = - params

    def apply_to_img(self, result):
        for key in result.get('img_fields', []):
            result[key] = ndi.rotate(result[key], self.params, self.axes, self.reshape, None, self.order, self.mode, self.val)

    def apply_to_seg(self, result):
        for key in result.get('seg_fields', []):
            result[key] = ndi.rotate(result[key], self.params, self.axes, self.reshape, None, 0, self.mode)

    def apply_to_det(self, result):
        for key in result.get('det_fields', []):
            rotated_bboxes = []
            for bbox in result[key]:
                # print(bbox)
                selected = [self.dim - i for i in self.axes] + [2 * self.dim - i for i in self.axes]
                # print(selected)
                x_min, y_min, x_max, y_max = bbox[selected]
                x = np.array([x_min, x_max, x_max, x_min]) - 0.5 * self.array_shape[self.axes[0]]
                y = np.array([y_min, y_min, y_max, y_max]) - 0.5 * self.array_shape[self.axes[1]]
                angle = np.deg2rad(self.params)
                x_t = np.cos(angle) * x + np.sin(angle) * y
                y_t = -np.sin(angle) * x + np.cos(angle) * y
                x_t = x_t + 0.5 * self.array_shape[self.axes[0]]
                y_t = y_t + 0.5 * self.array_shape[self.axes[1]]

                x_min, x_max = min(x_t), max(x_t)
                y_min, y_max = min(y_t), max(y_t)
                bbox[selected] = [x_min, y_min, x_max, y_max]

                rotated_bboxes.append(bbox)

            result[key] = clipBBoxes(self.dim, np.array(rotated_bboxes), self.image_shape)


def apply_transform(x, transform_matrix, fill_mode='nearest', cval=0.,
                    order=0, rows_idx=1, cols_idx=2):
    """Apply an affine transformation on each channel separately."""
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]

    # Reshape to (*, 0, 1)
    pattern = [el for el in range(x.ndim) if el != rows_idx and el != cols_idx]
    pattern += [rows_idx, cols_idx]
    inv_pattern = [pattern.index(el) for el in range(x.ndim)]
    x = x.transpose(pattern)
    x_shape = list(x.shape)
    x = x.reshape([-1] + x_shape[-2:])  # squash everything on the first axis

    # Apply the transformation on each channel, sequence, batch, ..
    for i in range(x.shape[0]):
        x[i] = ndi.interpolation.affine_transform(x[i], final_affine_matrix,
                                                  final_offset, order=order,
                                                  mode=fill_mode, cval=cval)
    x = x.reshape(x_shape)  # unsquash
    x = x.transpose(inv_pattern)
    return x


class RandomElasticDeformation(AugBase):
    def __init__(self, p,
                 num_control_points: Union[int, Tuple[int, int, int]] = 8,
                 max_displacement: float = 0.8):
        super().__init__()
        self.p = p
        self.num_control_points = num_control_points  # zyx order
        self.max_displacement = max_displacement  # zyx order
        self.SPLINE_ORDER = 3
        self.num_locked_borders = 2
        self.image_interpolation = 'linear'
        if self.image_interpolation == 'linear':
            self.Interpolator = sitk.sitkLinear
        else:
            raise NotImplementedError
        assert max_displacement <= 1.0
        self.tmp_params = None

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += "(p={}, num_control_points={}, max_displacement={})" \
            .format(self.p, self.num_control_points, self.max_displacement)
        return repr_str

    @property
    def canBackward(self):
        return True

    @staticmethod
    def parse_free_form_transform(transform, max_displacement):
        """Issue a warning is possible folding is detected."""
        coefficient_images = transform.GetCoefficientImages()
        grid_spacing = coefficient_images[0].GetSpacing()
        conflicts = np.array(max_displacement) > np.array(grid_spacing) / 2
        if np.any(conflicts):
            where, = np.where(conflicts)
            message = (
                'The maximum displacement is larger than the coarse grid'
                f' spacing for dimensions: {where.tolist()}, so folding may'
                ' occur. Choose fewer control points or a smaller'
                ' maximum displacement'
            )
            warnings.warn(message, RuntimeWarning)

    def _forward_params(self, result):
        self._init_params(result)
        grid_shape = self.to_tuple(self.num_control_points, self.dim)
        grid_spacing = self.image_shape / (np.array(grid_shape) - 1)
        max_displacement = self.max_displacement * grid_spacing / 2
        coarse_field = np.random.rand(*grid_shape, self.dim)  # [0, 1)
        coarse_field = 2 * coarse_field - 1  # [-1, 1)
        for d in range(self.dim):
            # [-max_displacement, max_displacement)
            coarse_field[..., d] *= max_displacement[d]

        # Set displacement to 0 at the borders
        for i in range(self.num_locked_borders):
            for d in range(self.dim):
                coarse_field[i, :] = 0
                coarse_field[-1 - i, :] = 0
                coarse_field[:, i] = 0
                coarse_field[:, -1 - i] = 0
                if self.dim == 3:
                    coarse_field[:, :, i] = 0
                    coarse_field[:, :, -1 - i] = 0
        self.params = coarse_field
        result[self.key_name] = self.params

    def _backward_params(self, result):
        self._init_params(result)
        params = result.pop(self.key_name, None)
        if params is not None:
            self.params = - params

    def elastic_transform(self, image: np.ndarray, isLabel=False):
        if not isLabel:
            Interpolator = self.Interpolator
        else:
            Interpolator = sitk.sitkNearestNeighbor

        axes = tuple(reversed(range(self.dim)))
        transformed_result = []
        for component in image:
            component = np.transpose(component, axes)

            sitkImage = sitk.GetImageFromArray(component, isVector=False)

            floating = reference = sitkImage
            grid_shape = self.to_tuple(self.num_control_points, self.dim)
            mesh_shape = [n - self.SPLINE_ORDER for n in grid_shape]
            bspline_transform = sitk.BSplineTransformInitializer(sitkImage, mesh_shape, self.SPLINE_ORDER)
            parameters = self.params.flatten(order='F').tolist()
            bspline_transform.SetParameters(parameters)
            self.parse_free_form_transform(bspline_transform, self.max_displacement)

            # medthod 1
            # https://github.com/fvisin/dataset_loaders/blob/master/dataset_loaders/data_augmentation.py
            displacement_filter = sitk.TransformToDisplacementFieldFilter()
            displacement_filter.SetReferenceImage(reference)
            displacement_field = displacement_filter.Execute(bspline_transform)
            self.tmp_params = sitk.GetArrayFromImage(displacement_field).transpose(axes + (self.dim,))

            # if self.dim == 3:
            #     result['RandomElasticDeformation_field'] = sitk.GetArrayFromImage(displacement_field).transpose((2, 1, 0, 3))

            warp_filter = sitk.WarpImageFilter()
            warp_filter.SetInterpolator(Interpolator)
            warp_filter.SetEdgePaddingValue(np.min(component).astype(np.double))
            image_warped = warp_filter.Execute(floating, displacement_field)
            image_warped = sitk.GetArrayFromImage(image_warped)
            image_warped = np.transpose(image_warped, axes)
            transformed_result.append(image_warped)

            # method 2
            # https://torchio.readthedocs.io/_modules/torchio/transforms/augmentation/spatial/random_elastic_deformation.html#RandomElasticDeformation
            # resampler = sitk.ResampleImageFilter()
            # resampler.SetReferenceImage(reference)
            # resampler.SetTransform(bspline_transform)
            # resampler.SetInterpolator(sitk.sitkLinear)
            # resampler.SetDefaultPixelValue(np.min(component).item())
            # resampler.SetOutputPixelType(sitk.sitkFloat32)
            # resampled = resampler.Execute(floating)
            # resampled = sitk.GetArrayFromImage(resampled)
            # resampled = np.transpose(resampled, (2, 1, 0))
            # resampled_result.append(resampled)
            # print(resampled.shape)
        image = np.stack(transformed_result, axis=0).astype(np.float32)
        return image

    def apply_to_img(self, result):
        for key in result.get('img_fields', []):
            result[key] = self.elastic_transform(result[key])

    def apply_to_seg(self, result):
        for key in result['seg_fields']:
            result[key] = self.elastic_transform(result[key], isLabel=True)

    def apply_to_det(self, result):
        for key in result['det_fields']:
            new_dets = np.copy(result[key])
            try:
                for det in new_dets:
                    # https://stackoverflow.com/questions/12935194/combinations-between-two-lists
                    clist = [list(range(i, 2 * self.dim, self.dim)) for i in range(self.dim)]
                    cord_idx = list(itertools.product(*clist))  # all corners
                    # print(cord_idx)
                    # print(det)
                    transformed_coords = []
                    for i, idx in enumerate(cord_idx):
                        voxel = np.int64(det[list(idx)]).reshape(self.dim, -1)
                        voxel = tuple(voxel[::-1].tolist())
                        # print(voxel)
                        transformed_coords.append(det[list(idx)] - self.tmp_params[voxel][0, ::-1])
                    transformed_coords = np.stack(transformed_coords, axis=0)
                    for i in range(self.dim):
                        det[i] = np.min(transformed_coords[:, i])
                        det[i + self.dim] = np.max(transformed_coords[:, i])
                result[key] = new_dets
            except Exception as e:
                print(self.name, result[key])
                print(self.name, result['history'])
                raise e


# -------------- Crop Patch --------------- #


class CropRandom(AugBase):
    """
    support segmentation, detection and classification tasks
    support 2D and 3D images
    """

    def __init__(self, patch_size=(128, 128), times=1):
        super().__init__()
        self.always = True
        self.patch_size = patch_size
        self.times = times

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += "(patch_size={})".format(self.patch_size)
        return repr_str

    @property
    def repeats(self):
        return self.times

    def _forward_params(self, result):
        self._init_params(result)
        # print(self.key_name, np.random.get_state()[1][0])
        assert self.dim == len(self.patch_size)
        assert all([self.image_shape[i] >= self.patch_size[i] for i in range(self.dim)]), self.image_shape

        start = tuple(map(lambda a, da: np.random.randint(0, a - da + 1), self.image_shape, self.patch_size))
        end = tuple(map(lambda a, b: a + b, start, self.patch_size))
        # 2d [(0, 1), (62, 63)]
        # 3d [(0, 1, 2), (62, 63, 64)]
        self.params = [start, end]
        result[self.key_name] = self.params
        # print(self.key_name, np.random.get_state()[1][0])

    def apply_to_img(self, result):
        for key in result.get('img_fields', []):
            start, end = self.params
            slices = (slice(None),) + tuple(map(slice, start, end))
            result[key] = result[key][slices]
            if key == 'img':
                result['img_shape'] = result[key].shape
            assert result[key].shape[1:] == self.patch_size, f'crop error! cropped shape is {result[key].shape}'

    def apply_to_seg(self, result):
        for key in result.get('seg_fields', []):
            start, end = self.params
            slices = (slice(None),) + tuple(map(slice, start, end))
            result[key] = result[key][slices]

    def apply_to_det(self, result):
        for key in result.get('det_fields', []):
            start, end = self.params
            result[key] = cropBBoxes(self.dim, result[key], start[::-1], end[::-1], dim_iou_thr=0.8)


class CropWeighted(CropRandom):
    def _forward_params(self, result):
        assert 'pixel_weight' in result.keys(), 'pixel_weight not in result keys'
        weights = result['pixel_weight'][0]
        assert weights.ndim in (2, 3)
        ctr_Nd = [np.arange(0, s) for s in weights.shape]  # zyx order
        ctr_Nd = np.meshgrid(*ctr_Nd, indexing='ij')

        indices = np.arange(len(weights.flat))
        index = np.random.choice(indices, 1, p=weights.flat / np.sum(weights))
        pos = tuple([c.flat[index] for c in ctr_Nd])
        print(pos)

        start = tuple([p - self.patch_size[i] // 2 for i, p in enumerate(pos)])
        end = tuple(map(lambda a, b: a + b, start, self.patch_size))
        self.params = [start, end]
        result[self.key_name] = self.params
        result['patch_size'] = self.patch_size


class CropCenter(CropRandom):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _forward_params(self, result):
        self._init_params(result)
        assert self.dim == len(self.patch_size)
        assert all([self.image_shape[i] >= self.patch_size[i] for i in range(self.dim)]), self.image_shape

        start = tuple(map(lambda a, da: a // 2 - da // 2, self.image_shape, self.patch_size))
        end = tuple(map(lambda a, b: a + b, start, self.patch_size))
        self.params = [start, end]
        result[self.key_name] = self.params
        result['patch_size'] = self.patch_size


class CropForeground(CropRandom):
    def __init__(self, border=12, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.border = border

    def _forward_params(self, result):
        self._init_params(result)
        assert self.dim == len(self.patch_size)
        assert all([self.image_shape[i] >= self.patch_size[i] for i in range(self.dim)]), self.image_shape

        assert 'gt_seg' in result.keys() or 'pseudo_mask' in result.keys()
        try:
            if 'gt_seg' in result.keys():
                foreground = result['gt_seg'][0]
            else:
                foreground = result['pseudo_mask'][0]
            objs = ndi.find_objects(ndi.label(foreground)[0])
            if len(objs):
                obj = random.choice(objs)
                patch_start_min = tuple([
                    min(max(obj[dim].stop + self.border - self.patch_size[dim], 0),
                        self.image_shape[dim] - self.patch_size[dim])
                    for dim in range(len(obj))])
                patch_start_max = tuple(
                    [min(max(obj[dim].start - self.border, 0), self.image_shape[dim] - self.patch_size[dim])
                     for dim in range(len(obj))])
                start = tuple(map(lambda a, da: randint(min(a, da), max(a, da) + 1), patch_start_min, patch_start_max))
                end = tuple(map(lambda a, b: a + b, start, self.patch_size))
                self.params = [start, end]
                result[self.key_name] = self.params
                result['patch_size'] = self.patch_size
            else:
                CropRandom._forward_params(self, result)
        except Exception as e:
            raise e
            # start = tuple(map(lambda a, da: np.random.randint(0, a - da + 1), self.image_shape, self.patch_size))
            # end = tuple(map(lambda a, b: a + b, start, self.patch_size))


class CropDet(CropRandom):
    def __init__(self, border=12, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.border = border

    def _forward_params(self, result):
        self._init_params(result)
        assert self.dim == len(self.patch_size)
        assert all([self.image_shape[i] >= self.patch_size[i] for i in range(self.dim)]), self.image_shape

        assert 'gt_det' in result.keys(), "it only used for detection tasks"
        try:
            selected_bbox_idx = min(self.current_repeat_idx % self.repeats, len(result['gt_det']) - 1)
            obj = result['gt_det'][selected_bbox_idx]
            self.current_repeat_idx += 1
            obj = [slice(obj[self.dim - i - 1], obj[2 * self.dim - i - 1]) for i in range(self.dim)]
            patch_start_min = tuple(
                [min(max(obj[dim].stop + self.border - self.patch_size[dim], 0),
                     self.image_shape[dim] - self.patch_size[dim])
                 for dim in range(len(obj))])
            patch_start_max = tuple(
                [min(max(obj[dim].start - self.border, 0),
                     self.image_shape[dim] - self.patch_size[dim])
                 for dim in range(len(obj))])
            start = tuple(map(lambda a, da: randint(min(a, da), max(a, da) + 1), patch_start_min, patch_start_max))
            end = tuple(map(lambda a, b: a + b, start, self.patch_size))
        except Exception as e:
            raise e
            # start = tuple(map(lambda a, da: np.random.randint(0, a - da + 1), self.image_shape, self.patch_size))
            # end = tuple(map(lambda a, b: a + b, start, self.patch_size))
        self.params = [start, end]
        result[self.key_name] = self.params
        result['patch_size'] = self.patch_size


class CropFirstDet(CropRandom):
    def __init__(self, patch_size=(128, 128), border=12):
        super().__init__(patch_size)
        self.border = border

    def _forward_params(self, result):
        self._init_params(result)
        assert self.dim == len(self.patch_size)
        assert all([self.image_shape[i] >= self.patch_size[i] for i in range(self.dim)]), self.image_shape

        assert 'gt_det' in result.keys(), "it only used for detection tasks"
        try:
            obj = result['gt_det'][0]
            obj = [slice(obj[self.dim - i - 1], obj[2 * self.dim - i - 1]) for i in range(self.dim)]
            patch_start_min = tuple(
                [min(max(obj[dim].stop + self.border - self.patch_size[dim], 0),
                     self.image_shape[dim] - self.patch_size[dim])
                 for dim in range(len(obj))])
            patch_start_max = tuple(
                [min(max(obj[dim].start - self.border, 0),
                     self.image_shape[dim] - self.patch_size[dim])
                 for dim in range(len(obj))])
            start = tuple(map(lambda a, da: randint(min(a, da), max(a, da) + 1), patch_start_min, patch_start_max))
            end = tuple(map(lambda a, b: a + b, start, self.patch_size))
        except Exception as e:
            raise e
            # start = tuple(map(lambda a, da: np.random.randint(0, a - da + 1), self.image_shape, self.patch_size))
            # end = tuple(map(lambda a, b: a + b, start, self.patch_size))
        self.params = [start, end]
        result[self.key_name] = self.params
        result['patch_size'] = self.patch_size


class CropFirstDetOnly(CropFirstDet):
    def apply_to_det(self, result):
        for key in result.get('det_fields', []):
            start, end = self.params
            result[key] = cropBBoxes(self.dim, result[key][:1, ...], start[::-1], end[::-1], dim_iou_thr=0.8)


class ForegroundPatches(AugBase):
    def __init__(self, patch_size=(128, 128), border=12, background=0):
        super().__init__()
        self.always = True
        self.border = border
        self.patch_size = patch_size
        self.background = background

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += "(border={}, patch_size={}, background={})".format(self.border, self.patch_size, self.background)
        return repr_str

    def _forward_params(self, result: dict):
        self._init_params(result)
        assert self.dim == len(self.patch_size)
        assert all([self.image_shape[i] >= self.patch_size[i] for i in range(self.dim)]), self.image_shape

        assert 'gt_seg' in result.keys() or 'pseudo_mask' in result.keys()
        try:
            if 'gt_seg' in result.keys():
                foreground = result['gt_seg'][0]
            else:
                foreground = result['pseudo_mask'][0]
            self.try_to_info(1)
            foreground, _ = ndi.label(foreground)
            objs = ndi.find_objects(foreground)
            del foreground
            gc.collect()
            print('Split into', len(objs), 'patches')
            self.try_to_info(2)
            anchors = []
            for obj in objs:
                patch_start_min = tuple([
                    min(max(obj[dim].stop + self.border - self.patch_size[dim], 0),
                        self.image_shape[dim] - self.patch_size[dim])
                    for dim in range(len(obj))])
                patch_start_max = tuple(
                    [min(max(obj[dim].start - self.border, 0), self.image_shape[dim] - self.patch_size[dim])
                     for dim in range(len(obj))])
                start = tuple(map(lambda a, da: randint(min(a, da), max(a, da) + 1), patch_start_min, patch_start_max))
                end = tuple(map(lambda a, b: a + b, start, self.patch_size))
                anchors.append(list(start + end))
            self.try_to_info(3)

            for i in range(self.background):
                start = tuple(map(lambda a, da: np.random.randint(0, a - da + 1), self.image_shape, self.patch_size))
                end = tuple(map(lambda a, b: a + b, start, self.patch_size))
                anchors.append(list(start + end))
        except Exception as e:
            raise e
            # start = tuple(map(lambda a, da: np.random.randint(0, a - da + 1), self.image_shape, self.patch_size))
            # end = tuple(map(lambda a, b: a + b, start, self.patch_size))
        self.params = np.array(anchors)
        result[self.key_name] = self.params

    def apply_to_img(self, result: dict):
        image = result.pop('img')

        patches = []
        for anchor in self.params:
            slices = tuple(slice(anchor[i], anchor[i + self.dim]) for i in range(self.dim))
            slices = (slice(None),) + slices
            patches.append(image[slices])

        result['patches_img'] = np.stack(patches, axis=0)

    def apply_to_seg(self, result: dict):
        for key in result.get('seg_fields', []):
            image = result.pop(key)

            labeled, num_objs = ndi.label(image[0])
            objs = ndi.find_objects(labeled)
            bboxes = objs2bboxes(objs)
            bboxes = np.concatenate([bboxes, np.arange(num_objs)[:, None]], axis=1)

            patches = []
            for anchor in self.params:
                self.try_to_info(4.1)
                slices = tuple(slice(anchor[i], anchor[i + self.dim]) for i in range(self.dim))
                slices = (slice(None),) + slices

                image_patch = image[slices]
                patch = np.zeros_like(image_patch)
                # print(patch.shape)
                self.try_to_info(4.2)

                start, end = anchor[:self.dim], anchor[self.dim:]
                keep_bboxes = cropBBoxes(self.dim, bboxes, start[::-1], end[::-1], dim_iou_thr=0.7)
                keep_objs = bboxes2objs(keep_bboxes)
                for obj, obj_idx in zip(keep_objs, keep_bboxes[:, -1]):
                    # print(obj)
                    keep_obj_patch = (labeled == (obj_idx + 1))[slices[1:]]
                    keep_obj_patch = np.repeat(keep_obj_patch[None, ...], image.shape[0], axis=0)
                    obj = (slice(None),) + tuple(obj)
                    patch[obj] = (keep_obj_patch * image_patch)[obj]
                patches.append(patch)
                self.try_to_info(5)

            result['patches_' + key] = np.stack(patches, axis=0)

    def apply_to_det(self, result: dict):
        for key in result.get('det_fields', []):
            print('\033[31m{}-Warning: Please use Crop instead!\033[0m'.format(self.__class__.__name__))
            patches_bboxes = []
            for p, anchor in enumerate(self.params):
                # np array
                start, end = anchor[:self.dim], anchor[self.dim:]
                cropped_bboxes = cropBBoxes(self.dim, result[key], start[::-1], end[::-1], dim_iou_thr=0.7)
                patches_bboxes.append(cropped_bboxes)
            result['patches_' + key] = patches_bboxes
