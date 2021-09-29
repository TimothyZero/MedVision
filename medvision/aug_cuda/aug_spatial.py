# -*- coding:utf-8 -*-
import math
import time
import warnings
import itertools
from typing import Union, Tuple
import torch
import torch.nn.functional as F
import numpy as np
import random
import SimpleITK as sitk
import scipy.ndimage as ndi

from .base import CudaAugBase
from .utils import cropBBoxes, clipBBoxes
from ..ops.cuda_fun_tools import affine_2d, affine_3d
from ..ops.cuda_fun_tools import apply_offset_2d, apply_offset_3d


class CudaResize(CudaAugBase):
    def __init__(self, spacing=None, scale=None, factor=None, order=1):
        super().__init__()
        self.always = True
        self.spacing = spacing
        self.scale = scale
        self.factor = factor
        self.order = order
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
        image = result['img']
        assert image.is_cuda, 'image should be cuda'
        assert image.ndim == self.dim + 1, 'image should be channel, **dim'
        device = image.device

        if self.dim == 2:
            cuda_fun = affine_2d
        else:
            cuda_fun = affine_3d

        if not all([i == 1.0 for i in self.params]):
            index = torch.FloatTensor([0])
            center = torch.FloatTensor([i / 2 for i in list(self.image_shape[::-1])])  # dhw => xyz
            shape = torch.FloatTensor(list(self.image_shape[::-1]))  # dhw => xyz
            if self.dim == 2:
                angles = torch.FloatTensor([0])
            else:
                angles = torch.FloatTensor([0, 0, 0])

            rois = torch.cat([index, center, shape, angles]).unsqueeze(0).to(device)
            out_size = tuple([int(i * p) for i, p in zip(self.image_shape, self.params)])
            spatial_scale = 1
            aligned = True
            order = self.order

            image = cuda_fun(
                image.unsqueeze(0),
                rois,
                out_size,
                spatial_scale,
                1,
                aligned,
                order
            ).squeeze(0)
        result['img'] = image
        result['img_shape'] = tuple(image.shape)
        # result['img_spacing'] = tuple(np.array(result['img_spacing']) / np.array(self.params))

    def apply_to_seg(self, result):
        for key in result.get('seg_fields', []):
            image = result[key]
            assert image.is_cuda, 'image should be cuda'
            assert image.ndim == self.dim + 1, 'image should be channel, **dim'
            device = image.device

            if self.dim == 2:
                cuda_fun = affine_2d
            else:
                cuda_fun = affine_3d

            if not all([i == 1.0 for i in self.params]):
                index = torch.FloatTensor([0])
                center = torch.FloatTensor([i / 2 for i in list(self.image_shape[::-1])])  # dhw => xyz
                shape = torch.FloatTensor(list(self.image_shape[::-1]))  # dhw => xyz
                if self.dim == 2:
                    angles = torch.FloatTensor([0])
                else:
                    angles = torch.FloatTensor([0, 0, 0])

                rois = torch.cat([index, center, shape, angles]).unsqueeze(0).to(device)
                out_size = tuple([int(i * p) for i, p in zip(self.image_shape, self.params)])
                spatial_scale = 1
                aligned = True
                order = 0

                image = cuda_fun(
                    image.float().unsqueeze(0),
                    rois,
                    out_size,
                    spatial_scale,
                    1,
                    aligned,
                    order
                ).squeeze(0).int()
            result[key] = image

    def apply_to_det(self, result):
        for key in result.get('det_fields', []):
            if not all([i == 1.0 for i in self.params]):
                scale = torch.from_numpy(np.hstack(self.params[::-1] * 2)).to(result[key].device)
                result[key][:, :2 * self.dim] = result[key][:, :2 * self.dim] * scale


class CudaPad(CudaAugBase):
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

        # 3d [(32, 32), (12, 12), (14, 14)]
        # => [32, 32, 12, 12, 14, 14]
        self.params = [a for i in diff for a in i]
        result[self.key_name] = self.params

    def _backward_params(self, result):
        params = self._init_params(result)
        if params:
            # 2d [(32, 32), (32, 32)]
            # 3d [(32, 32), (32, 32), (32, 32)]
            self.params = params

    def apply_to_img(self, result):
        img = result['img']
        if self.isForwarding:
            if not all([i == 0 for i in self.params]):
                # https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
                # pad: left, right, top, bottom, front, back
                img = F.pad(img, self.params[::-1], mode=self.mode, value=self.val)
                assert all([i <= j for i, j in zip(self.size, img.shape[1:])]), \
                    f"image shape is {img.shape} while required is {self.size}"
        else:
            slices = tuple([slice(lp, shape - rp) for (lp, rp), shape in zip(self.params, self.image_shape)])
            slices = (slice(None),) + slices
            img = img[slices]

        result['img'] = img
        result['img_shape'] = tuple(img.shape)

    def apply_to_seg(self, result):
        for key in result.get('seg_fields', []):
            if self.isForwarding:
                result[key] = F.pad(result[key], self.params[::-1], mode=self.mode, value=self.val)
            else:
                slices = tuple([slice(lp, shape - rp) for (lp, rp), shape in zip(self.params, self.image_shape)])
                slices = (slice(None),) + slices
                result[key] = result[key][slices]

    def apply_to_det(self, result):
        # 2d [(32, 42), (33, 43)]
        # forward  [33, 32]
        # backward [-33, -32]
        ops = 1 if self.isForwarding else -1
        offsets = [i * ops for i in self.params[::2]][::-1] * 2
        # print(offsets)
        for key in result.get('det_fields', []):
            offsets = torch.from_numpy(np.array(offsets)).to(result[key].device)
            result[key][:, :2 * self.dim] = result[key][:, :2 * self.dim] + offsets


class CudaRandomAffine(CudaAugBase):
    def __init__(self,
                 p,
                 scale: Union[float, list, tuple],  # one axis only
                 shift: Union[float, list, tuple],
                 rotate: Union[float, list, tuple],  # degree 0-180
                 sampling_ratio=1,
                 order=1):
        """
        rotate: The direction of vector rotation is counterclockwise if θ is positive.
        """
        super(CudaRandomAffine, self).__init__()
        self.p = p
        self.scale = scale
        self.shift = shift
        self.rotate = rotate  # temp: only rotate on xy plane
        self.sampling_ratio = sampling_ratio
        self.order = order

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(dim={}, scale={}, shift={}, rotate={})'.format(self.dim, self.scale, self.shift, self.rotate)
        return repr_str

    def _forward_params(self, result):
        self._init_params(result)
        _scales = [self.get_range(self.scale, 1), ] * self.dim
        _shifts = [self.get_range(self.shift, 0) for _ in range(self.dim)]
        _rotate = [math.pi * self.get_range(self.rotate, 0) / 180]
        self.params = {
            "_scales": _scales,
            "_shifts": _shifts,
            "_rotate": _rotate
        }
        result[self.key_name] = self.params

    def _backward_params(self, result: dict):
        self._init_params(result)
        params = result.get(self.key_name, None)
        if params:
            _scales = params["_scales"]
            _shifts = params["_shifts"]
            _rotate = params["_rotate"]
            self.params = {
                "_scales": 1 / _scales,
                "_shifts": - _shifts,
                "_rotate": - _rotate
            }

    def apply_to_img(self, result: dict):
        """
        image : 1, c, ***
        """
        if self.dim == 2:
            cuda_fun = affine_2d
        else:
            cuda_fun = affine_3d

        image = result['img']
        assert image.is_cuda, 'image should be cuda'
        assert image.ndim == self.dim + 1, 'image should be channel, **dim'
        device = image.device

        _scales = torch.FloatTensor(self.params["_scales"])
        _shifts = torch.FloatTensor(self.params["_shifts"])
        _rotate = torch.FloatTensor(self.params["_rotate"])

        index = torch.FloatTensor([0])
        center = torch.FloatTensor([i / 2 for i in list(self.image_shape[::-1])])  # dhw => xyz
        shape = torch.FloatTensor(list(self.image_shape[::-1]))  # dhw => xyz
        if self.dim == 2:
            angles = torch.FloatTensor([_rotate])
        else:
            angles = torch.FloatTensor([0, 0, _rotate])

        rois = torch.cat([index, center - shape * _shifts / _scales, shape / _scales, angles]).unsqueeze(0).to(device)
        out_size = image.shape[1:]
        spatial_scale = 1.0
        aligned = True
        order = self.order

        img = cuda_fun(
            image.unsqueeze(0),
            rois,
            out_size,
            spatial_scale,
            self.sampling_ratio,
            aligned,
            order
        ).squeeze(0)
        result['img'] = img
        result['img_shape'] = tuple(img.shape)

    def apply_to_cls(self, result: dict):
        pass

    def apply_to_seg(self, result: dict):
        """
        gt_seg : 1, c, ***
        """
        if self.dim == 2:
            cuda_fun = affine_2d
        else:
            cuda_fun = affine_3d

        for key in result.get('seg_fields', []):
            image = result[key]
            assert image.is_cuda, 'seg should be cuda'
            assert image.ndim == self.dim + 1, 'seg should be channel, **dim'
            device = image.device

            _scales = torch.FloatTensor(self.params["_scales"])
            _shifts = torch.FloatTensor(self.params["_shifts"])
            _rotate = torch.FloatTensor(self.params["_rotate"])

            index = torch.FloatTensor([0])
            center = torch.FloatTensor([i / 2 for i in list(self.image_shape[::-1])])  # dhw => xyz
            shape = torch.FloatTensor(list(self.image_shape[::-1]))  # dhw => xyz
            if self.dim == 2:
                angles = torch.FloatTensor([_rotate])
            else:
                angles = torch.FloatTensor([0, 0, _rotate])

            rois = torch.cat([index, center - shape * _shifts / _scales, shape / _scales, angles]).unsqueeze(0).to(
                device)
            out_size = image.shape[1:]
            spatial_scale = 1.0
            aligned = True
            order = 0

            seg = cuda_fun(
                image.float().unsqueeze(0),
                rois,
                out_size,
                spatial_scale,
                self.sampling_ratio,
                aligned,
                order
            ).squeeze(0).int()
            result[key] = seg

    def apply_to_det(self, result: dict):
        for key in result.get('det_fields', []):
            bboxes = result[key].cpu().numpy()
            expanded = np.ones((bboxes.shape[0], 2 * self.dim + 2))
            expanded[:, :self.dim] = bboxes[:, :self.dim]
            expanded[:, self.dim + 1:2 * self.dim + 1] = bboxes[:, self.dim:2 * self.dim]

            # print(self.params)
            _scales = self.params["_scales"]
            _shifts = self.params["_shifts"]
            _rotate = self.params["_rotate"]
            _shape = self.image_shape[::-1]  # already xyz order
            if self.dim == 3:
                ShiftM = np.array([
                    [1, 0, 0, _shape[0] * _shifts[0] / _scales[0]],
                    [0, 1, 0, _shape[1] * _shifts[1] / _scales[1]],
                    [0, 0, 1, _shape[2] * _shifts[2] / _scales[2]],
                    [0, 0, 0, 1],
                ])
                ScaleM = np.array([
                    [_scales[0], 0, 0, - _shape[0] / 2 * (_scales[0] - 1)],
                    [0, _scales[1], 0, - _shape[1] / 2 * (_scales[1] - 1)],
                    [0, 0, _scales[2], - _shape[2] / 2 * (_scales[2] - 1)],
                    [0, 0, 0, 1],
                ])
                expanded[:, :self.dim + 1] = np.matmul(np.matmul(ScaleM, ShiftM), expanded[:, :self.dim + 1].T).T
                expanded[:, self.dim + 1:] = np.matmul(np.matmul(ScaleM, ShiftM), expanded[:, self.dim + 1:].T).T
                bboxes = np.concatenate([expanded[:, [0, 1, 2, 4, 5, 6]], bboxes[:, -2:]], axis=1)
            else:
                ShiftM = np.array([
                    [1, 0, _shape[0] * _shifts[0] / _scales[0]],
                    [0, 1, _shape[1] * _shifts[1] / _scales[1]],
                    [0, 0, 1],
                ])
                ScaleM = np.array([
                    [_scales[0], 0, - _shape[0] / 2 * (_scales[0] - 1)],
                    [0, _scales[1], - _shape[1] / 2 * (_scales[1] - 1)],
                    [0, 0, 1],
                ])
                expanded[:, :self.dim + 1] = np.matmul(np.matmul(ScaleM, ShiftM), expanded[:, :self.dim + 1].T).T
                expanded[:, self.dim + 1:] = np.matmul(np.matmul(ScaleM, ShiftM), expanded[:, self.dim + 1:].T).T
                # print(expanded)
                bboxes = np.concatenate([expanded[:, [0, 1, 3, 4]], bboxes[:, -2:]], axis=1)

            rotated_bboxes = []
            for bbox in bboxes:
                # print(bbox)
                if self.dim == 2:
                    selected = [0, 1, 2, 3]
                else:
                    selected = [0, 1, 3, 4]

                # while doing roi align rotated, we rotate bbox by theta with image fixed.
                # actually, we fill one by one in output roi array with image rotated by theta,
                # so to rotate matrix is the same with .cuda script but the angle is reversed
                angle = - _rotate[0]

                x_min, y_min, x_max, y_max = bbox[selected]
                x = np.array([x_min, x_max, x_max, x_min]) - 0.5 * _shape[0]
                y = np.array([y_min, y_min, y_max, y_max]) - 0.5 * _shape[1]

                x_t = np.cos(angle) * x - np.sin(angle) * y
                y_t = np.sin(angle) * x + np.cos(angle) * y
                x_t = x_t + 0.5 * _shape[0]
                y_t = y_t + 0.5 * _shape[1]

                x_min, x_max = min(x_t), max(x_t)
                y_min, y_max = min(y_t), max(y_t)
                bbox[selected] = [x_min, y_min, x_max, y_max]

                rotated_bboxes.append(bbox)
            valid_bboxes = clipBBoxes(self.dim, np.array(rotated_bboxes), self.image_shape)
            result[key] = torch.from_numpy(valid_bboxes).to(result[key].device)


class CudaRandomScale(CudaRandomAffine):
    def __init__(self,
                 p,
                 scale: Union[float, list, tuple],
                 sampling_ratio=1
                 ):
        super().__init__(p, scale=scale, shift=0, rotate=0, sampling_ratio=sampling_ratio)


class CudaRandomShift(CudaRandomAffine):
    def __init__(self,
                 p,
                 shift: Union[float, list, tuple],
                 sampling_ratio=1
                 ):
        super().__init__(p, scale=0, shift=shift, rotate=0, sampling_ratio=sampling_ratio)


class CudaRandomRotate(CudaRandomAffine):
    def __init__(self,
                 p,
                 rotate: Union[float, list, tuple],
                 sampling_ratio=1
                 ):
        super().__init__(p, scale=0, shift=0, rotate=rotate, sampling_ratio=sampling_ratio)


class CudaRandomElasticDeformation(CudaAugBase):
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
        self.tmp_params = None  # remember to clear at beginning

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
        self.tmp_params = None

        self._init_params(result)
        grid_shape = self.to_tuple(self.num_control_points, self.dim)
        grid_spacing = self.image_shape / (np.array(grid_shape) - 1)
        max_displacement = self.max_displacement * grid_spacing / 2
        coarse_field = np.random.rand(*grid_shape, self.dim).astype(np.float32)  # [0, 1)
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
        self.tmp_params = None

        self._init_params(result)
        params = result.pop(self.key_name, None)
        if params is not None:
            self.params = - params

    def elastic_transform_itk(self, image: torch.Tensor, order=1):
        """ slower but work """
        tic = time.time()

        if self.tmp_params is None:
            axes = tuple(reversed(range(self.dim)))

            component = image[0].permute(*axes).cpu().numpy().astype(np.float32)

            sitkImage = sitk.GetImageFromArray(component, isVector=False)

            reference = sitkImage
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
            offset = sitk.GetArrayFromImage(displacement_field).transpose((self.dim,) + axes).astype(np.float32)
            offset = torch.from_numpy(offset)
            self.tmp_params = offset
        toc = time.time()
        if order == 0:
            image = image.to(self.img_type)
        if self.dim == 2:
            image = apply_offset_2d(
                image.unsqueeze(0),
                self.tmp_params.unsqueeze(0),
                order=order).squeeze(0)
        elif self.dim == 3:
            image = apply_offset_3d(
                image.unsqueeze(0),
                self.tmp_params.unsqueeze(0),
                order=order).squeeze(0)
        if order == 0:
            image = image.int()
        toc2 = time.time()
        # print("toc - tic", toc - tic)
        # print("toc2 - toc", toc2 - toc)
        # np.save(f'offset_{self.dim}d.npy', self.tmp_params)
        # np.save(f'grid_offset_{self.dim}d.npy', self.params)
        return image

    def elastic_transform(self, image: torch.Tensor, order=1):
        """ cubic interpolation to get offset"""
        tic = time.time()
        if self.tmp_params is None:
            # make first dimension is offset on each dim, e.g. 2
            # / 2 to smooth
            grid_offset = torch.from_numpy(self.params).permute(self.dim, *range(self.dim)) / 2
            image_shape = grid_offset.shape[1:]
            index = torch.FloatTensor([0])
            center = torch.FloatTensor([i / 2 for i in list(image_shape[::-1])])
            shape = torch.FloatTensor(list(image_shape[::-1])) - 3
            if self.dim == 2:
                angles = torch.FloatTensor([0])
            else:
                angles = torch.FloatTensor([0, 0, 0])

            rois = torch.cat([index, center, shape, angles]).unsqueeze(0).to('cuda')
            out_size = self.image_shape
            spatial_scale = 1
            aligned = True

            if self.dim == 2:
                cuda_fun = affine_2d
            else:
                cuda_fun = affine_3d

            offset = cuda_fun(
                grid_offset.cuda().unsqueeze(0),
                rois,
                out_size,
                spatial_scale,
                1,
                aligned,
                3
            ).squeeze(0).cuda()
            self.tmp_params = offset

        toc = time.time()
        if order == 0:
            image = image.to(self.img_type)
        if self.dim == 2:
            image = apply_offset_2d(
                image.unsqueeze(0),
                self.tmp_params.unsqueeze(0),
                order=order
            ).squeeze(0)
        elif self.dim == 3:
            image = apply_offset_3d(
                image.unsqueeze(0),
                self.tmp_params.unsqueeze(0),
                order=order
            ).squeeze(0)
        if order == 0:
            image = image.int()
        toc2 = time.time()
        # print("toc - tic", toc - tic)
        # print("toc2 - toc", toc2 - toc)
        # np.save(f'fast_offset_{self.dim}d.npy', self.tmp_params.cpu().numpy())
        # np.save(f'fast_grid_offset_{self.dim}d.npy', self.params)
        return image

    def apply_to_img(self, result):
        result['img'] = self.elastic_transform(result['img'])

    def apply_to_seg(self, result):
        for key in result.get('seg_fields', []):
            result[key] = self.elastic_transform(result[key], order=0).int()

    def apply_to_det(self, result):
        for key in result.get('det_fields', []):
            new_dets = np.copy(result[key].cpu().numpy())
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
                        # print(2, voxel)
                        voxel_offset = self.tmp_params[(slice(None),) + voxel].cpu().numpy()[::-1, 0]
                        transformed_coords.append(det[list(idx)] - voxel_offset)
                    transformed_coords = np.stack(transformed_coords, axis=0)
                    for i in range(self.dim):
                        det[i] = np.min(transformed_coords[:, i])
                        det[i + self.dim] = np.max(transformed_coords[:, i])
                result[key] = torch.from_numpy(new_dets).to(result[key].device)
            except Exception as e:
                print(self.name, result[key])
                print(self.name, result['history'])
                raise e


class CudaRandomFlip(CudaAugBase):
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
        flipped = [i + 1 for i, f in enumerate(self.params) if f == -1]
        if len(flipped):
            result['img'] = torch.flip(result['img'], flipped)

    def apply_to_seg(self, result):
        flipped = [i + 1 for i, f in enumerate(self.params) if f == -1]
        if len(flipped):
            for key in result.get('seg_fields', []):
                result[key] = torch.flip(result[key], flipped)

    def apply_to_det(self, result):
        for key in result.get('det_fields', []):
            bboxes = np.array(result[key].cpu().numpy())
            for i, tag in enumerate(self.params[::-1]):  # xyz
                if tag == -1:
                    bboxes[:, i] = self.image_shape[- i - 1] - bboxes[:, i] - 1
                    bboxes[:, i + self.dim] = self.image_shape[- i - 1] - bboxes[:, i + self.dim] - 1
                    bboxes[:, [i, i + self.dim]] = bboxes[:, [i + self.dim, i]]
            result[key] = torch.from_numpy(bboxes).to(result[key].device)


class CudaCropRandom(CudaAugBase):
    def __init__(self,
                 patch_size=(128, 128),
                 times=1):
        super(CudaCropRandom, self).__init__()
        self.always = True
        self.dim = len(patch_size)
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

    def _backward_params(self, result: dict):
        raise NotImplementedError

    def apply_to_img(self, result):
        img = result['img']
        start, end = self.params
        slices = (slice(None),) + tuple(map(slice, start, end))
        cropped = img[slices]
        result['img'] = cropped
        result['img_shape'] = cropped.shape
        assert cropped.shape[1:] == self.patch_size, \
            f'Cropped shape is {cropped.shape}, pre shape is {img.shape}, patch shape is {self.patch_size}'

    def apply_to_seg(self, result):
        for key in result.get('seg_fields', []):
            start, end = self.params
            slices = (slice(None),) + tuple(map(slice, start, end))
            result[key] = result[key][slices]

    def apply_to_det(self, result):
        for key in result.get('det_fields', []):
            start, end = self.params
            bboxes = result[key].cpu().numpy()
            bboxes = cropBBoxes(self.dim, bboxes, start[::-1], end[::-1], dim_iou_thr=0.8)
            result[key] = torch.from_numpy(bboxes).to(result[key].device)


class CudaCropCenter(CudaCropRandom):
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


class CudaCropForeground(CudaCropRandom):
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
                raise NotImplementedError

            objs = ndi.find_objects(ndi.label(foreground.cpu().numpy())[0])
            if len(objs):
                obj = random.choice(objs)
                patch_start_min = tuple([
                    min(max(obj[dim].stop + self.border - self.patch_size[dim], 0),
                        self.image_shape[dim] - self.patch_size[dim])
                    for dim in range(len(obj))])
                patch_start_max = tuple(
                    [min(max(obj[dim].start - self.border, 0), self.image_shape[dim] - self.patch_size[dim])
                     for dim in range(len(obj))])
                start = tuple(map(lambda a, da: np.random.randint(min(a, da), max(a, da) + 1), patch_start_min, patch_start_max))
                end = tuple(map(lambda a, b: a + b, start, self.patch_size))
                self.params = [start, end]
                result[self.key_name] = self.params
            else:
                CudaCropRandom._forward_params(self, result)
        except Exception as e:
            raise e
            # start = tuple(map(lambda a, da: np.random.randint(0, a - da + 1), self.image_shape, self.patch_size))
            # end = tuple(map(lambda a, b: a + b, start, self.patch_size))


class CudaCropFirstDet(CudaCropRandom):
    def __init__(self, patch_size=(128, 128), border=12, *args, **kwargs):
        super().__init__(patch_size, *args, **kwargs)
        self.border = border

    def _forward_params(self, result):
        super(CudaCropFirstDet, self)._init_params(result)
        assert self.dim == len(self.patch_size)
        assert all([self.image_shape[i] >= self.patch_size[i] for i in range(self.dim)]), self.image_shape

        assert 'gt_det' in result.keys(), "it only used for detection tasks"
        if len(result['gt_det']) > 0:
            try:
                obj = result['gt_det'][0].cpu().numpy()
                obj = [slice(obj[self.dim - i - 1], obj[2 * self.dim - i - 1]) for i in range(self.dim)]
                patch_start_min = tuple(
                    [min(max(obj[dim].stop + self.border - self.patch_size[dim], 0),
                         self.image_shape[dim] - self.patch_size[dim])
                     for dim in range(len(obj))])
                patch_start_max = tuple(
                    [min(max(obj[dim].start - self.border, 0),
                         self.image_shape[dim] - self.patch_size[dim])
                     for dim in range(len(obj))])
                start = tuple(
                    map(lambda a, da: np.random.randint(min(a, da), max(a, da) + 1), patch_start_min, patch_start_max))
                end = tuple(map(lambda a, b: a + b, start, self.patch_size))
            except Exception as e:
                raise e
                # start = tuple(map(lambda a, da: np.random.randint(0, a - da + 1), self.image_shape, self.patch_size))
                # end = tuple(map(lambda a, b: a + b, start, self.patch_size))
            self.params = [start, end]
            result[self.key_name] = self.params
        else:
            CudaCropRandom._forward_params(self, result)


class CudaCropDet(CudaCropRandom):
    def __init__(self, patch_size=(128, 128), border=12, *args, **kwargs):
        super().__init__(patch_size, *args, **kwargs)
        self.border = border

    def _forward_params(self, result):
        super(CudaCropDet, self)._init_params(result)
        assert self.dim == len(self.patch_size)
        assert all([self.image_shape[i] >= self.patch_size[i] for i in range(self.dim)]), self.image_shape

        assert 'gt_det' in result.keys(), "it only used for detection tasks"
        if len(result['gt_det']) > 0:
            try:
                bboxes = result['gt_det'].cpu().numpy()
                obj = bboxes[np.random.randint(0, len(bboxes))]
                obj = [slice(obj[self.dim - i - 1], obj[2 * self.dim - i - 1]) for i in range(self.dim)]
                patch_start_min = tuple(
                    [min(max(obj[dim].stop + self.border - self.patch_size[dim], 0),
                         self.image_shape[dim] - self.patch_size[dim])
                     for dim in range(len(obj))])
                patch_start_max = tuple(
                    [min(max(obj[dim].start - self.border, 0),
                         self.image_shape[dim] - self.patch_size[dim])
                     for dim in range(len(obj))])
                start = tuple(
                    map(lambda a, da: np.random.randint(min(a, da), max(a, da) + 1), patch_start_min, patch_start_max))
                end = tuple(map(lambda a, b: a + b, start, self.patch_size))
            except Exception as e:
                raise e
                # start = tuple(map(lambda a, da: np.random.randint(0, a - da + 1), self.image_shape, self.patch_size))
                # end = tuple(map(lambda a, b: a + b, start, self.patch_size))
            self.params = [start, end]
            result[self.key_name] = self.params
        else:
            CudaCropRandom._forward_params(self, result)


class CudaCropFirstDetOnly(CudaCropFirstDet):
    def apply_to_det(self, result):
        for key in result.get('det_fields', []):
            start, end = self.params
            bboxes = result[key].cpu().numpy()
            bboxes = cropBBoxes(self.dim, bboxes[:1, ...], start[::-1], end[::-1], dim_iou_thr=0.8)
            result[key] = torch.from_numpy(bboxes).to(result[key].device)




class CudaCropRandomWithAffine(CudaAugBase):
    def __init__(self,
                 patch_size,
                 scale: Union[float, list, tuple],  # one axis only
                 shift: Union[float, list, tuple],
                 rotate: Union[float, list, tuple],  # degree 0-180
                 sampling_ratio=1,
                 order=1,
                 times=1):
        super(CudaCropRandomWithAffine, self).__init__()
        self.always = True
        self.dim = len(patch_size)
        self.patch_size = patch_size
        self.scale = scale
        self.shift = shift
        self.rotate = rotate  # temp: only rotate on xy plane
        self.sampling_ratio = sampling_ratio
        self.order = order
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

        start, end = self._forward_crop_params(result)
        _scales, _shifts, _rotate = self._forward_affine_params(result)

        self.params = self.params = {
            "start": start[::-1],  # xyz
            "end": end[::-1],
            "_scales": _scales,
            "_shifts": _shifts,
            "_rotate": _rotate
        }
        result[self.key_name] = self.params

    def _forward_crop_params(self, result):
        assert self.dim == len(self.patch_size)
        assert all([self.image_shape[i] >= self.patch_size[i] for i in range(self.dim)]), self.image_shape

        start = tuple(map(lambda a, da: np.random.randint(0, a - da + 1), self.image_shape, self.patch_size))
        end = tuple(map(lambda a, b: a + b, start, self.patch_size))
        return start, end

    def _forward_affine_params(self, result):
        _scales = [self.get_range(self.scale, 1), ] * self.dim
        _shifts = [self.get_range(self.shift, 0) for _ in range(self.dim)]
        _rotate = [math.pi * self.get_range(self.rotate, 0) / 180]
        return _scales, _shifts, _rotate

    def apply_to_img(self, result: dict):
        if self.dim == 2:
            cuda_fun = affine_2d
        else:
            cuda_fun = affine_3d

        image = result['img']
        assert image.is_cuda, 'image should be cuda'
        assert image.ndim == self.dim + 1, 'image should be channel, **dim'
        device = image.device

        start = torch.FloatTensor(self.params["start"])
        end = torch.FloatTensor(self.params["end"])
        _scales = torch.FloatTensor(self.params["_scales"])
        _shifts = torch.FloatTensor(self.params["_shifts"])
        _rotate = torch.FloatTensor(self.params["_rotate"])

        index = torch.FloatTensor([0])
        center = (start + end) / 2
        shape = end - start
        if self.dim == 2:
            angles = torch.FloatTensor([_rotate])
        else:
            angles = torch.FloatTensor([0, 0, _rotate])

        rois = torch.cat([index, center - shape * _shifts / _scales, shape / _scales, angles]).unsqueeze(0).to(device)
        out_size = self.patch_size
        spatial_scale = 1.0
        aligned = True
        order = self.order

        img = cuda_fun(
            image.unsqueeze(0),
            rois,
            out_size,
            spatial_scale,
            self.sampling_ratio,
            aligned,
            order
        ).squeeze(0)
        result['img'] = img
        result['img_shape'] = tuple(img.shape)

    def apply_to_seg(self, result: dict):
        if self.dim == 2:
            cuda_fun = affine_2d
        else:
            cuda_fun = affine_3d

        for key in result.get('seg_fields', []):
            image = result[key]
            assert image.is_cuda, 'image should be cuda'
            assert image.ndim == self.dim + 1, 'image should be channel, **dim'
            device = image.device

            start = torch.FloatTensor(self.params["start"])
            end = torch.FloatTensor(self.params["end"])
            _scales = torch.FloatTensor(self.params["_scales"])
            _shifts = torch.FloatTensor(self.params["_shifts"])
            _rotate = torch.FloatTensor(self.params["_rotate"])

            index = torch.FloatTensor([0])
            center = (start + end) / 2
            shape = end - start
            if self.dim == 2:
                angles = torch.FloatTensor([_rotate])
            else:
                angles = torch.FloatTensor([0, 0, _rotate])

            rois = torch.cat([index, center - shape * _shifts / _scales, shape / _scales, angles]).unsqueeze(0).to(device)
            out_size = self.patch_size
            spatial_scale = 1.0
            aligned = True
            order = 0

            img = cuda_fun(
                image.float().unsqueeze(0),
                rois,
                out_size,
                spatial_scale,
                self.sampling_ratio,
                aligned,
                order
            ).squeeze(0).int()
            result[key] = img

    def apply_to_det(self, result: dict):
        for key in result.get('det_fields', []):
            bboxes = result[key].cpu().numpy()
            expanded = np.ones((bboxes.shape[0], 2 * self.dim + 2))
            expanded[:, :self.dim] = bboxes[:, :self.dim]
            expanded[:, self.dim + 1:2 * self.dim + 1] = bboxes[:, self.dim:2 * self.dim]

            start = torch.FloatTensor(self.params["start"])
            end = torch.FloatTensor(self.params["end"])
            _scales = self.params["_scales"]
            _shifts = self.params["_shifts"]
            _rotate = self.params["_rotate"]
            _shape = self.patch_size[::-1]
            if self.dim == 3:
                ShiftM = np.array([
                    [1, 0, 0, _shape[0] * _shifts[0] / _scales[0] - start[0]],
                    [0, 1, 0, _shape[1] * _shifts[1] / _scales[1] - start[1]],
                    [0, 0, 1, _shape[2] * _shifts[2] / _scales[2] - start[2]],
                    [0, 0, 0, 1],
                ])
                ScaleM = np.array([
                    [_scales[0], 0, 0, - _shape[0] / 2 * (_scales[0] - 1)],
                    [0, _scales[1], 0, - _shape[1] / 2 * (_scales[1] - 1)],
                    [0, 0, _scales[2], - _shape[2] / 2 * (_scales[2] - 1)],
                    [0, 0, 0, 1],
                ])
                expanded[:, :self.dim + 1] = np.matmul(np.matmul(ScaleM, ShiftM), expanded[:, :self.dim + 1].T).T
                expanded[:, self.dim + 1:] = np.matmul(np.matmul(ScaleM, ShiftM), expanded[:, self.dim + 1:].T).T
                bboxes = np.concatenate([expanded[:, [0, 1, 2, 4, 5, 6]], bboxes[:, -2:]], axis=1)
            else:
                ShiftM = np.array([
                    [1, 0, _shape[0] * _shifts[0] / _scales[0] - start[0]],
                    [0, 1, _shape[1] * _shifts[1] / _scales[1] - start[1]],
                    [0, 0, 1],
                ])
                ScaleM = np.array([
                    [_scales[0], 0, - _shape[0] / 2 * (_scales[0] - 1)],
                    [0, _scales[1], - _shape[1] / 2 * (_scales[1] - 1)],
                    [0, 0, 1],
                ])
                expanded[:, :self.dim + 1] = np.matmul(np.matmul(ScaleM, ShiftM), expanded[:, :self.dim + 1].T).T
                expanded[:, self.dim + 1:] = np.matmul(np.matmul(ScaleM, ShiftM), expanded[:, self.dim + 1:].T).T
                # print(expanded)
                bboxes = np.concatenate([expanded[:, [0, 1, 3, 4]], bboxes[:, -2:]], axis=1)

            rotated_bboxes = []
            for bbox in bboxes:
                # print(bbox)
                if self.dim == 2:
                    selected = [0, 1, 2, 3]
                else:
                    selected = [0, 1, 3, 4]
                angle = - _rotate[0]

                x_min, y_min, x_max, y_max = bbox[selected]
                x = np.array([x_min, x_max, x_max, x_min]) - 0.5 * _shape[0]
                y = np.array([y_min, y_min, y_max, y_max]) - 0.5 * _shape[1]

                x_t = np.cos(angle) * x - np.sin(angle) * y
                y_t = np.sin(angle) * x + np.cos(angle) * y
                x_t = x_t + 0.5 * _shape[0]
                y_t = y_t + 0.5 * _shape[1]

                x_min, x_max = min(x_t), max(x_t)
                y_min, y_max = min(y_t), max(y_t)
                bbox[selected] = [x_min, y_min, x_max, y_max]

                rotated_bboxes.append(bbox)
            valid_bboxes = clipBBoxes(self.dim, np.array(rotated_bboxes), result['img'].shape[1:])
            result[key] = torch.from_numpy(valid_bboxes).to(result[key].device)


class CudaCropCenterWithAffine(CudaCropRandomWithAffine):
    def _forward_crop_params(self, result):
        assert self.dim == len(self.patch_size)
        assert all([self.image_shape[i] >= self.patch_size[i] for i in range(self.dim)]), self.image_shape

        start = tuple(map(lambda a, da: a // 2 - da // 2, self.image_shape, self.patch_size))
        end = tuple(map(lambda a, b: a + b, start, self.patch_size))
        return start, end


class CudaCropFirstDetWithAffine(CudaCropRandomWithAffine):
    def __init__(self, border=12, *args, **kwargs):
        self.border = border
        super(CudaCropFirstDetWithAffine, self).__init__(*args, **kwargs)

    def _forward_crop_params(self, result):
        assert self.dim == len(self.patch_size)
        assert all([self.image_shape[i] >= self.patch_size[i] for i in range(self.dim)]), self.image_shape

        assert 'gt_det' in result.keys(), "it only used for detection tasks"
        if len(result['gt_det']) > 0:
            try:
                obj = result['gt_det'][0].cpu().numpy()
                obj = [slice(obj[self.dim - i - 1], obj[2 * self.dim - i - 1]) for i in range(self.dim)]
                patch_start_min = tuple(
                    [min(max(obj[dim].stop + self.border - self.patch_size[dim], 0),
                         self.image_shape[dim] - self.patch_size[dim])
                     for dim in range(len(obj))])
                patch_start_max = tuple(
                    [min(max(obj[dim].start - self.border, 0),
                         self.image_shape[dim] - self.patch_size[dim])
                     for dim in range(len(obj))])
                start = tuple(
                    map(lambda a, da: np.random.randint(min(a, da), max(a, da) + 1), patch_start_min, patch_start_max))
                end = tuple(map(lambda a, b: a + b, start, self.patch_size))
                return start, end
            except Exception as e:
                raise e
                # start = tuple(map(lambda a, da: np.random.randint(0, a - da + 1), self.image_shape, self.patch_size))
                # end = tuple(map(lambda a, b: a + b, start, self.patch_size))
        else:
            return super(CudaCropFirstDetWithAffine, self)._forward_crop_params(result)

