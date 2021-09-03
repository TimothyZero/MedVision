import math
import time
import warnings
import itertools
from typing import Union, Iterable, Tuple, List
import torch
import torch.nn.functional as F
import numpy as np
import random
import SimpleITK as sitk
import scipy.ndimage as ndi

from ..aug_cuda.cuda_fun_tools import affine_2d, affine_3d
from ..aug_cuda.cuda_fun_tools import apply_offset_2d, apply_offset_3d
from .base import BatchCudaAugBase


class BatchCudaRandomFlip(BatchCudaAugBase):
    def __init__(self, p,
                 axes: list = None):
        super(BatchCudaRandomFlip, self).__init__()
        self.p = p
        self.axes = axes  # not used
        self.flip_p = 0.5

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(p={}, axes={})'.format(self.p, self.axes)
        return repr_str

    @property
    def canBackward(self):
        return True

    def _forward_params(self, result: dict):
        self._init_params(result)
        batch_params = []
        for i in range(self.batch):
            batch_params.append(random.choices([-1, 1],
                                               weights=[self.flip_p, 1 - self.flip_p],
                                               k=self.dim))
        self.params = batch_params
        result[self.key_name] = tuple(self.params)

    def _backward_params(self, result: dict):
        self._init_params(result)
        params = result.pop(self.key_name, None)
        if params:
            # 2d [-1, 1]
            # 3d [1, -1, 1]
            self.params = params

    def apply_to_img(self, result):
        for b in range(self.batch):
            # b, c, d, h, w
            flipped = [i + 1 for i, f in enumerate(self.params[b]) if f == -1]
            if len(flipped):
                result['img'][b] = torch.flip(result['img'][b], flipped)

    def apply_to_seg(self, result):
        for b in range(self.batch):
            flipped = [i + 1 for i, f in enumerate(self.params[b]) if f == -1]
            if len(flipped):
                for key in result.get('seg_fields', []):
                    result[key][b] = torch.flip(result[key][b], flipped)

    def apply_to_det(self, result):
        for key in result.get('det_fields', []):
            raise NotImplementedError


class BatchCudaResize(BatchCudaAugBase):
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
        params = super()._backward_params(result)
        if params:
            # 2d (2.0, 2.0)
            # 3d (2.0, 2.0, 2.0)
            self.params = tuple(1 / np.array(params))

    def apply_to_img(self, result):
        image = result['img']
        assert image.is_cuda, 'image should be cuda'
        assert image.ndim == self.dim + 2, 'image should be batch, channel, **dim'
        device = image.device

        if self.dim == 2:
            cuda_fun = affine_2d
        else:
            cuda_fun = affine_3d

        if not all([i == 1.0 for i in self.params]):
            index = torch.FloatTensor(list(range(self.batch))).unsqueeze(-1)
            center = torch.FloatTensor([i / 2 for i in list(self.image_shape[::-1])])  # dhw => xyz
            shape = torch.FloatTensor(list(self.image_shape[::-1]))  # dhw => xyz
            if self.dim == 2:
                angles = torch.FloatTensor([0])
            else:
                angles = torch.FloatTensor([0, 0, 0])

            bboxes = torch.cat([center, shape, angles]).unsqueeze(0).repeat(self.batch, 1)
            rois = torch.cat([index, bboxes], dim=-1).to(device)
            out_size = tuple([int(i * p) for i, p in zip(self.image_shape, self.params)])
            spatial_scale = 1
            aligned = True
            order = self.order

            image = cuda_fun(
                image,
                rois,
                out_size,
                spatial_scale,
                1,
                aligned,
                order
            )
        result['img'] = image
        # result['img_spacing'] = tuple(np.array(result['img_spacing']) / np.array(self.params))

    def apply_to_seg(self, result):
        for key in result.get('seg_fields', []):
            image = result[key]
            assert image.is_cuda, 'image should be cuda'
            assert image.ndim == self.dim + 2, 'image should be batch, channel, **dim'
            device = image.device

            if self.dim == 2:
                cuda_fun = affine_2d
            else:
                cuda_fun = affine_3d

            if not all([i == 1.0 for i in self.params]):
                index = torch.FloatTensor(list(range(self.batch))).unsqueeze(-1)
                center = torch.FloatTensor([i / 2 for i in list(self.image_shape[::-1])])  # dhw => xyz
                shape = torch.FloatTensor(list(self.image_shape[::-1]))  # dhw => xyz
                if self.dim == 2:
                    angles = torch.FloatTensor([0])
                else:
                    angles = torch.FloatTensor([0, 0, 0])

                bboxes = torch.cat([center, shape, angles]).unsqueeze(0).repeat(self.batch, 1)
                rois = torch.cat([index, bboxes], dim=-1).to(device)
                out_size = tuple([int(i * p) for i, p in zip(self.image_shape, self.params)])
                spatial_scale = 1
                aligned = True
                order = 0

                image = cuda_fun(
                    image.float(),
                    rois,
                    out_size,
                    spatial_scale,
                    1,
                    aligned,
                    order
                ).int()
            result[key] = image

    def apply_to_det(self, result):
        for key in result.get('det_fields', []):
            raise NotImplementedError
            # if not all([i == 1.0 for i in self.params]):
            #     result[key][..., :2 * self.dim] = result[key][..., :2 * self.dim] * np.hstack(self.params[::-1] * 2)


class BatchCudaRandomElasticDeformationFast(BatchCudaAugBase):
    def __init__(self, p,
                 num_control_points: Union[int, Tuple[int, int, int]] = 8,
                 max_displacement: float = 0.8,
                 order=3):
        super().__init__()
        self.p = p
        self.num_control_points = num_control_points  # zyx order
        self.max_displacement = max_displacement  # zyx order
        self.INTER_ORDER = order
        self.num_locked_borders = 2
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

    def _forward_params(self, result):
        self.tmp_params = None

        self._init_params(result)
        batch_coarse_field = []
        for i in range(self.batch):
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
            batch_coarse_field.append(coarse_field)
        self.params = np.array(batch_coarse_field)
        result[self.key_name] = self.params

    def _backward_params(self, result):
        self.tmp_params = None

        self._init_params(result)
        params = result.pop(self.key_name, None)
        if params is not None:
            self.params = - params

    def elastic_transform(self, image: torch.Tensor, order=1):
        tic = time.time()
        device = image.device
        if self.tmp_params is None:
            # make first dimension is offset on each dim, e.g. 2
            # / 2 to smooth
            grid_offset = torch.from_numpy(self.params).permute(0, self.dim + 1, *range(1, self.dim + 1)) / 2
            image_shape = grid_offset.shape[2:]
            index = torch.FloatTensor(list(range(self.batch))).unsqueeze(-1)
            center = torch.FloatTensor([i / 2 for i in list(image_shape[::-1])])
            shape = torch.FloatTensor(list(image_shape[::-1])) - (2 * self.num_locked_borders - 1)
            if self.dim == 2:
                angles = torch.FloatTensor([0])
            else:
                angles = torch.FloatTensor([0, 0, 0])

            bboxes = torch.cat([center, shape, angles]).unsqueeze(0).repeat(self.batch, 1)
            rois = torch.cat([index, bboxes], dim=-1).to(device)
            out_size = self.image_shape
            spatial_scale = 1
            aligned = True

            if self.dim == 2:
                cuda_fun = affine_2d
            else:
                cuda_fun = affine_3d

            offset = cuda_fun(
                grid_offset.cuda(),
                rois,
                out_size,
                spatial_scale,
                1,
                aligned,
                self.INTER_ORDER
            )
            self.tmp_params = offset

        toc = time.time()
        # if order == 0:
        #     image = image.to(self.img_type)
        if self.dim == 2:
            image = apply_offset_2d(image, self.tmp_params, order=order)
        elif self.dim == 3:
            image = apply_offset_3d(image, self.tmp_params, order=order)
        # if order == 0:
        #     image = image.int()
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
            raise NotImplementedError
            # new_dets = np.copy(result[key])
            # try:
            #     for det in new_dets:
            #         # https://stackoverflow.com/questions/12935194/combinations-between-two-lists
            #         clist = [list(range(i, 2 * self.dim, self.dim)) for i in range(self.dim)]
            #         cord_idx = list(itertools.product(*clist))  # all corners
            #         # print(cord_idx)
            #         # print(det)
            #         transformed_coords = []
            #         for i, idx in enumerate(cord_idx):
            #             voxel = np.int64(det[list(idx)]).reshape(self.dim, -1)
            #             voxel = tuple(voxel[::-1].tolist())
            #             # print(voxel)
            #             voxel_offset = self.tmp_params[(slice(None),) + voxel].cpu().numpy()[::-1, 0]
            #             transformed_coords.append(det[list(idx)] - voxel_offset)
            #         transformed_coords = np.stack(transformed_coords, axis=0)
            #         for i in range(self.dim):
            #             det[i] = np.min(transformed_coords[:, i])
            #             det[i + self.dim] = np.max(transformed_coords[:, i])
            #     result[key] = new_dets
            # except Exception as e:
            #     print(self.name, result[key])
            #     print(self.name, result['history'])
            #     raise e


class BatchCudaRandomAffine(BatchCudaAugBase):
    def __init__(self,
                 p,
                 scale: Union[float, list, tuple],  # one axis only
                 shift: Union[float, list, tuple],
                 rotate: Union[float, list, tuple],  # degree 0-180
                 sampling_ratio=1,
                 order=1):
        super(BatchCudaRandomAffine, self).__init__()
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
        scales, shifts, rotate = [], [], []
        for i in range(self.batch):
            scales.append([self.get_range(self.scale, 1), ] * self.dim)
            shifts.append([self.get_range(self.shift, 0) for _ in range(self.dim)])
            rotate.append([math.pi * self.get_range(self.rotate, 0) / 180])
        self.params = {
            "scales": np.array(scales),
            "shifts": np.array(shifts),
            "rotate": np.array(rotate),
        }
        result[self.key_name] = self.params

    def _backward_params(self, result: dict):
        self._init_params(result)
        params = result.get(self.key_name, None)
        if params:
            scales = params["scales"]
            shifts = params["shifts"]
            rotate = params["rotate"]
            self.params = {
                "scales": 1 / scales,
                "shifts": - shifts,
                "rotate": - rotate
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
        assert image.ndim == self.dim + 2, 'image should be batch, channel, **dim'
        device = image.device

        scales = torch.FloatTensor(self.params["scales"])
        shifts = torch.FloatTensor(self.params["shifts"])
        rotate = torch.FloatTensor(self.params["rotate"])

        index = torch.FloatTensor(list(range(self.batch))).unsqueeze(-1)
        center = torch.FloatTensor([i / 2 for i in list(self.image_shape[::-1])])  # dhw => xyz
        shape = torch.FloatTensor(list(self.image_shape[::-1]))  # dhw => xyz
        if self.dim == 2:
            angles = rotate
        else:
            angles = torch.cat([torch.zeros_like(rotate), torch.zeros_like(rotate), rotate], dim=1)

        rois = torch.cat([index, center - shape * shifts / scales, shape / scales, angles], dim=-1).to(device)
        out_size = image.shape[2:]
        spatial_scale = 1.0
        aligned = True
        order = self.order

        img = cuda_fun(
            image,
            rois,
            out_size,
            spatial_scale,
            self.sampling_ratio,
            aligned,
            order
        )
        result['img'] = img

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
            assert image.is_cuda, 'image should be cuda'
            assert image.ndim == self.dim + 2, 'image should be batch, channel, **dim'
            device = image.device

            scales = torch.FloatTensor(self.params["scales"])
            shifts = torch.FloatTensor(self.params["shifts"])
            rotate = torch.FloatTensor(self.params["rotate"])

            index = torch.FloatTensor(list(range(self.batch))).unsqueeze(-1)
            center = torch.FloatTensor([i / 2 for i in list(self.image_shape[::-1])])  # dhw => xyz
            shape = torch.FloatTensor(list(self.image_shape[::-1]))  # dhw => xyz
            if self.dim == 2:
                angles = rotate
            else:
                angles = torch.cat([torch.zeros_like(rotate), torch.zeros_like(rotate), rotate], dim=1)

            rois = torch.cat([index, center - shape * shifts / scales, shape / scales, angles], dim=-1).to(device)
            out_size = image.shape[2:]
            spatial_scale = 1.0
            aligned = True
            order = 0

            seg = cuda_fun(
                image.float(),
                rois,
                out_size,
                spatial_scale,
                self.sampling_ratio,
                aligned,
                order
            ).int()
            result[key] = seg

    def apply_to_det(self, result: dict):
        for key in result.get('det_fields', []):
            raise NotImplementedError
            # bboxes = result[key]
            # expanded = np.ones((bboxes.shape[0], 2 * self.dim + 2))
            # expanded[:, :self.dim] = bboxes[:, :self.dim]
            # expanded[:, self.dim + 1:2 * self.dim + 1] = bboxes[:, self.dim:2 * self.dim]
            #
            # # print(self.params)
            # _scales = self.params["_scales"]
            # _shifts = self.params["_shifts"]
            # _rotate = self.params["_rotate"]
            # _shape = self.image_shape[::-1]  # already xyz order
            # if self.dim == 3:
            #     # print(expanded)
            #     ShiftM = np.array([
            #         [1, 0, 0, _shape[0] * _shifts[0] / _scales[0]],
            #         [0, 1, 0, _shape[1] * _shifts[1] / _scales[1]],
            #         [0, 0, 1, _shape[2] * _shifts[2] / _scales[2]],
            #         [0, 0, 0, 1],
            #     ])
            #     ScaleM = np.array([
            #         [_scales[0], 0, 0, - _shape[0] / 2 * (_scales[0] - 1)],
            #         [0, _scales[1], 0, - _shape[1] / 2 * (_scales[1] - 1)],
            #         [0, 0, _scales[2], - _shape[2] / 2 * (_scales[2] - 1)],
            #         [0, 0, 0, 1],
            #     ])
            #     expanded[:, :self.dim + 1] = np.matmul(np.matmul(ScaleM, ShiftM), expanded[:, :self.dim + 1].T).T
            #     expanded[:, self.dim + 1:] = np.matmul(np.matmul(ScaleM, ShiftM), expanded[:, self.dim + 1:].T).T
            #     # print(expanded)
            #     # RotateM = np.array([
            #     #     [ math.cos(_rotate[0]), math.sin(_rotate[0]), 0, 0.5 * _shape[0] - 0.5 * _shape[0] * math.cos(_rotate[0]) - 0.5 * _shape[1] * math.sin(_rotate[0])],
            #     #     [-math.sin(_rotate[0]), math.cos(_rotate[0]), 0, 0.5 * _shape[1] - 0.5 * _shape[1] * math.cos(_rotate[0]) + 0.5 * _shape[0] * math.sin(_rotate[0])],
            #     #     [0, 0, 1, 0],
            #     #     [0, 0, 0, 1],
            #     # ])
            #     result[key] = np.concatenate([expanded[:, [0, 1, 2, 4, 5, 6]], bboxes[:, -2:]], axis=1)
            # else:
            #     ShiftM = np.array([
            #         [1, 0, _shape[0] * _shifts[0] / _scales[0]],
            #         [0, 1, _shape[1] * _shifts[1] / _scales[1]],
            #         [0, 0, 1],
            #     ])
            #     ScaleM = np.array([
            #         [_scales[0], 0, - _shape[0] / 2 * (_scales[0] - 1)],
            #         [0, _scales[1], - _shape[1] / 2 * (_scales[1] - 1)],
            #         [0, 0, 1],
            #     ])
            #     expanded[:, :self.dim + 1] = np.matmul(np.matmul(ScaleM, ShiftM), expanded[:, :self.dim + 1].T).T
            #     expanded[:, self.dim + 1:] = np.matmul(np.matmul(ScaleM, ShiftM), expanded[:, self.dim + 1:].T).T
            #     # print(expanded)
            #     result[key] = np.concatenate([expanded[:, [0, 1, 3, 4]], bboxes[:, -2:]], axis=1)
            #
            # rotated_bboxes = []
            # for bbox in result[key]:
            #     # print(bbox)
            #     if self.dim == 2:
            #         selected = [0, 1, 2, 3]
            #     else:
            #         selected = [0, 1, 3, 4]
            #     # print(selected)
            #     x_min, y_min, x_max, y_max = bbox[selected]
            #     x = np.array([x_min, x_max, x_max, x_min]) - 0.5 * _shape[0]
            #     y = np.array([y_min, y_min, y_max, y_max]) - 0.5 * _shape[1]
            #     angle = _rotate[0]
            #     x_t = np.cos(angle) * x - np.sin(angle) * y
            #     y_t = np.sin(angle) * x + np.cos(angle) * y
            #     x_t = x_t + 0.5 * _shape[0]
            #     y_t = y_t + 0.5 * _shape[1]
            #
            #     x_min, x_max = min(x_t), max(x_t)
            #     y_min, y_max = min(y_t), max(y_t)
            #     bbox[selected] = [x_min, y_min, x_max, y_max]
            #
            #     rotated_bboxes.append(bbox)
            # result[key] = clipBBoxes(self.dim, np.array(rotated_bboxes), self.image_shape)


class BatchCudaRandomScale(BatchCudaRandomAffine):
    def __init__(self,
                 p,
                 scale: Union[float, list, tuple],
                 sampling_ratio=1
                 ):
        super().__init__(p, scale=scale, shift=0, rotate=0, sampling_ratio=sampling_ratio)


class BatchCudaRandomShift(BatchCudaRandomAffine):
    def __init__(self,
                 p,
                 shift: Union[float, list, tuple],
                 sampling_ratio=1
                 ):
        super().__init__(p, scale=0, shift=shift, rotate=0, sampling_ratio=sampling_ratio)


class BatchCudaRandomRotate(BatchCudaRandomAffine):
    def __init__(self,
                 p,
                 rotate: Union[float, list, tuple],
                 sampling_ratio=1
                 ):
        super().__init__(p, scale=0, shift=0, rotate=rotate, sampling_ratio=sampling_ratio)


class BatchCudaCropRandomWithAffine(BatchCudaAugBase):
    def __init__(self,
                 patch_size,
                 scale: Union[float, list, tuple],  # one axis only
                 shift: Union[float, list, tuple],
                 rotate: Union[float, list, tuple],  # degree 0-180
                 sampling_ratio=1,
                 order=1,
                 times=1):
        super(BatchCudaCropRandomWithAffine, self).__init__()
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

    @property
    def canBackward(self):
        return False

    def _forward_params(self, result):
        self._init_params(result)
        # print(self.key_name, np.random.get_state()[1][0])
        assert self.dim == len(self.patch_size)
        assert all([self.image_shape[i] >= self.patch_size[i] for i in range(self.dim)]), self.image_shape

        batch_start, batch_end = self._forward_crop_params(result)

        batch_scales, batch_shifts, batch_rotate = self._forward_affine_params(result)

        self.params = self.params = {
            "start": batch_start,  # xyz
            "end": batch_end,
            "scales": batch_scales,
            "shifts": batch_shifts,
            "rotate": batch_rotate
        }
        result[self.key_name] = self.params

    def _forward_affine_params(self, result: dict):
        batch_scales, batch_shifts, batch_rotate = [], [], []
        for i in range(self.batch):
            batch_scales.append([self.get_range(self.scale, 1), ] * self.dim)
            batch_shifts.append([self.get_range(self.shift, 0) for _ in range(self.dim)])
            batch_rotate.append([math.pi * self.get_range(self.rotate, 0) / 180])
        batch_scales = np.array(batch_scales)
        batch_shifts = np.array(batch_shifts)
        batch_rotate = np.array(batch_rotate)
        return batch_scales, batch_shifts, batch_rotate

    def _forward_crop_params(self, result: dict):
        batch_start, batch_end = [], []
        for i in range(self.batch):
            start = tuple(map(lambda a, da: np.random.randint(0, a - da + 1), self.image_shape, self.patch_size))
            end = tuple(map(lambda a, b: a + b, start, self.patch_size))
            batch_start.append(start[::-1])
            batch_end.append(end[::-1])
        batch_start = np.array(batch_start)
        batch_end = np.array(batch_end)
        return batch_start, batch_end

    def apply_to_img(self, result: dict):
        if self.dim == 2:
            cuda_fun = affine_2d
        else:
            cuda_fun = affine_3d

        image = result['img']
        assert image.is_cuda, 'image should be cuda'
        assert image.ndim == self.dim + 2, 'image should be batch, channel, **dim'
        device = image.device

        start = torch.FloatTensor(self.params["start"])
        end = torch.FloatTensor(self.params["end"])
        scales = torch.FloatTensor(self.params["scales"])
        shifts = torch.FloatTensor(self.params["shifts"])
        rotate = torch.FloatTensor(self.params["rotate"])

        index = torch.FloatTensor(list(range(self.batch))).unsqueeze(-1)
        center = (start + end) / 2
        shape = end - start
        if self.dim == 2:
            angles = rotate
        else:
            angles = torch.cat([torch.zeros_like(rotate), torch.zeros_like(rotate), rotate], dim=1)

        rois = torch.cat([index, center - shape * shifts / scales, shape / scales, angles], dim=-1).to(device)
        out_size = self.patch_size
        spatial_scale = 1.0
        aligned = True
        order = self.order

        img = cuda_fun(
            image,
            rois,
            out_size,
            spatial_scale,
            self.sampling_ratio,
            aligned,
            order
        )
        result['img'] = img

    def apply_to_seg(self, result: dict):
        if self.dim == 2:
            cuda_fun = affine_2d
        else:
            cuda_fun = affine_3d

        for key in result.get('seg_fields', []):
            image = result[key]
            assert image.is_cuda, 'image should be cuda'
            assert image.ndim == self.dim + 2, 'image should be batch, channel, **dim'
            device = image.device

            start = torch.FloatTensor(self.params["start"])
            end = torch.FloatTensor(self.params["end"])
            scales = torch.FloatTensor(self.params["scales"])
            shifts = torch.FloatTensor(self.params["shifts"])
            rotate = torch.FloatTensor(self.params["rotate"])

            index = torch.FloatTensor(list(range(self.batch))).unsqueeze(-1)
            center = (start + end) / 2
            shape = end - start
            if self.dim == 2:
                angles = rotate
            else:
                angles = torch.cat([torch.zeros_like(rotate), torch.zeros_like(rotate), rotate], dim=1)

            rois = torch.cat([index, center - shape * shifts / scales, shape / scales, angles], dim=-1).to(device)
            out_size = self.patch_size
            spatial_scale = 1.0
            aligned = True
            order = 0

            img = cuda_fun(
                image.float(),
                rois,
                out_size,
                spatial_scale,
                self.sampling_ratio,
                aligned,
                order
            ).int()
            result[key] = img

    def apply_to_det(self, result: dict):
        for key in result.get('det_fields', []):
            raise NotImplementedError
            # bboxes = result[key]
            # expanded = np.ones((bboxes.shape[0], 2 * self.dim + 2))
            # expanded[:, :self.dim] = bboxes[:, :self.dim]
            # expanded[:, self.dim + 1:2 * self.dim + 1] = bboxes[:, self.dim:2 * self.dim]
            #
            # start = torch.FloatTensor(self.params["start"])
            # end = torch.FloatTensor(self.params["end"])
            # _scales = self.params["_scales"]
            # _shifts = self.params["_shifts"]
            # _rotate = self.params["_rotate"]
            # _shape = self.patch_size[::-1]
            # if self.dim == 3:
            #     # print(expanded)
            #     ShiftM = np.array([
            #         [1, 0, 0, _shape[0] * _shifts[0] / _scales[0] - start[0]],
            #         [0, 1, 0, _shape[1] * _shifts[1] / _scales[1] - start[1]],
            #         [0, 0, 1, _shape[2] * _shifts[2] / _scales[2] - start[2]],
            #         [0, 0, 0, 1],
            #     ])
            #     ScaleM = np.array([
            #         [_scales[0], 0, 0, - _shape[0] / 2 * (_scales[0] - 1)],
            #         [0, _scales[1], 0, - _shape[1] / 2 * (_scales[1] - 1)],
            #         [0, 0, _scales[2], - _shape[2] / 2 * (_scales[2] - 1)],
            #         [0, 0, 0, 1],
            #     ])
            #     expanded[:, :self.dim + 1] = np.matmul(np.matmul(ScaleM, ShiftM), expanded[:, :self.dim + 1].T).T
            #     expanded[:, self.dim + 1:] = np.matmul(np.matmul(ScaleM, ShiftM), expanded[:, self.dim + 1:].T).T
            #     # print(expanded)
            #     # RotateM = np.array([
            #     #     [ math.cos(_rotate[0]), math.sin(_rotate[0]), 0, 0.5 * _shape[0] - 0.5 * _shape[0] * math.cos(_rotate[0]) - 0.5 * _shape[1] * math.sin(_rotate[0])],
            #     #     [-math.sin(_rotate[0]), math.cos(_rotate[0]), 0, 0.5 * _shape[1] - 0.5 * _shape[1] * math.cos(_rotate[0]) + 0.5 * _shape[0] * math.sin(_rotate[0])],
            #     #     [0, 0, 1, 0],
            #     #     [0, 0, 0, 1],
            #     # ])
            #     result[key] = np.concatenate([expanded[:, [0, 1, 2, 4, 5, 6]], bboxes[:, -2:]], axis=1)
            # else:
            #     ShiftM = np.array([
            #         [1, 0, _shape[0] * _shifts[0] / _scales[0] - start[0]],
            #         [0, 1, _shape[1] * _shifts[1] / _scales[1] - start[1]],
            #         [0, 0, 1],
            #     ])
            #     ScaleM = np.array([
            #         [_scales[0], 0, - _shape[0] / 2 * (_scales[0] - 1)],
            #         [0, _scales[1], - _shape[1] / 2 * (_scales[1] - 1)],
            #         [0, 0, 1],
            #     ])
            #     expanded[:, :self.dim + 1] = np.matmul(np.matmul(ScaleM, ShiftM), expanded[:, :self.dim + 1].T).T
            #     expanded[:, self.dim + 1:] = np.matmul(np.matmul(ScaleM, ShiftM), expanded[:, self.dim + 1:].T).T
            #     # print(expanded)
            #     result[key] = np.concatenate([expanded[:, [0, 1, 3, 4]], bboxes[:, -2:]], axis=1)
            #
            # rotated_bboxes = []
            # for bbox in result[key]:
            #     # print(bbox)
            #     if self.dim == 2:
            #         selected = [0, 1, 2, 3]
            #     else:
            #         selected = [0, 1, 3, 4]
            #     # print(selected)
            #     x_min, y_min, x_max, y_max = bbox[selected]
            #     x = np.array([x_min, x_max, x_max, x_min]) - 0.5 * _shape[0]
            #     y = np.array([y_min, y_min, y_max, y_max]) - 0.5 * _shape[1]
            #     angle = _rotate[0]
            #     x_t = np.cos(angle) * x - np.sin(angle) * y
            #     y_t = np.sin(angle) * x + np.cos(angle) * y
            #     x_t = x_t + 0.5 * _shape[0]
            #     y_t = y_t + 0.5 * _shape[1]
            #
            #     x_min, x_max = min(x_t), max(x_t)
            #     y_min, y_max = min(y_t), max(y_t)
            #     bbox[selected] = [x_min, y_min, x_max, y_max]
            #
            #     rotated_bboxes.append(bbox)
            # result[key] = clipBBoxes(self.dim, np.array(rotated_bboxes), self.patch_size)


class BatchCudaCropCenterWithAffine(BatchCudaCropRandomWithAffine):
    def _forward_crop_params(self, result: dict):
        batch_start, batch_end = [], []
        for i in range(self.batch):
            start = tuple(map(lambda a, da: a // 2 - da // 2, self.image_shape, self.patch_size))
            end = tuple(map(lambda a, b: a + b, start, self.patch_size))
            batch_start.append(start[::-1])
            batch_end.append(end[::-1])
        batch_start = np.array(batch_start)
        batch_end = np.array(batch_end)
        return batch_start, batch_end


class BatchCudaCropForegroundWithAffine(BatchCudaCropRandomWithAffine):
    def __init__(self, border=12, **kwargs):
        super(BatchCudaCropForegroundWithAffine, self).__init__(**kwargs)
        self.border = border

    def _forward_crop_params(self, result: dict):
        assert 'gt_seg' in result.keys() or 'pseudo_mask' in result.keys()

        batch_start, batch_end = [], []
        for i in range(self.batch):
            try:
                if 'gt_seg' in result.keys():
                    foreground = result['gt_seg'][i, 0]
                else:
                    foreground = result['pseudo_mask'][i, 0]
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
                    start = tuple(
                        map(lambda a, da: np.random.randint(min(a, da), max(a, da) + 1), patch_start_min, patch_start_max))
                    end = tuple(map(lambda a, b: a + b, start, self.patch_size))
                else:
                    start = tuple(map(lambda a, da: np.random.randint(0, a - da + 1), self.image_shape, self.patch_size))
                    end = tuple(map(lambda a, b: a + b, start, self.patch_size))
            except Exception as e:
                raise e

            batch_start.append(start[::-1])
            batch_end.append(end[::-1])
        batch_start = np.array(batch_start)
        batch_end = np.array(batch_end)
        return batch_start, batch_end


class BatchCudaCropDetWithAffine(BatchCudaCropRandomWithAffine):
    def __init__(self, border=12, **kwargs):
        super(BatchCudaCropDetWithAffine, self).__init__(**kwargs)
        self.border = border

    def _forward_crop_params(self, result: dict):
        assert 'gt_det' in result.keys(), "it only used for detection tasks"

        batch_start, batch_end = [], []
        for i in range(self.batch):
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
                start = tuple(map(lambda a, da: np.random.randint(min(a, da), max(a, da) + 1), patch_start_min, patch_start_max))
                end = tuple(map(lambda a, b: a + b, start, self.patch_size))
            except Exception as e:
                raise e

            batch_start.append(start[::-1])
            batch_end.append(end[::-1])
        batch_start = np.array(batch_start)
        batch_end = np.array(batch_end)
        return batch_start, batch_end


class BatchCudaCropFirstDetWithAffine(BatchCudaCropRandomWithAffine):
    def __init__(self, border=12, **kwargs):
        super(BatchCudaCropFirstDetWithAffine, self).__init__(**kwargs)
        self.border = border

    def _forward_crop_params(self, result: dict):
        assert 'gt_det' in result.keys(), "it only used for detection tasks"

        batch_start, batch_end = [], []
        for i in range(self.batch):
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
                start = tuple(map(lambda a, da: np.random.randint(min(a, da), max(a, da) + 1), patch_start_min, patch_start_max))
                end = tuple(map(lambda a, b: a + b, start, self.patch_size))
            except Exception as e:
                raise e

            batch_start.append(start[::-1])
            batch_end.append(end[::-1])
        batch_start = np.array(batch_start)
        batch_end = np.array(batch_end)
        return batch_start, batch_end