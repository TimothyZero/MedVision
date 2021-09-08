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

import gc
from copy import deepcopy
import numpy as np

from .base import AugBase
from .aug_spatial import Pad, Resize
from .utils import cropBBoxes, padBBoxes, nmsNd_numpy


class MultiScale(AugBase):
    def __init__(self, scales):
        super().__init__()
        self.always = True
        self.scales = scales.copy()
        self.tmp_scales = scales.copy()
        assert isinstance(scales, (list, tuple))

    @property
    def canBackward(self):
        return True

    @property
    def repeats(self):
        return len(self.scales)

    def _forward_params(self, result):
        self._init_params(result)
        # after a whole repeats of transform, self.tmp_img_scales will be a empty list
        # recover it from a copy of self.img_scales before new data
        if len(self.tmp_scales) == 0:
            self.tmp_scales = self.scales.copy()
        img_scale = self.tmp_scales.pop(0)

        img_shape = self.image_shape
        new_shape = np.array(img_scale)
        new_shape = np.where(new_shape == -1, img_shape, new_shape)
        scales = tuple(new_shape / img_shape)
        self.params = scales
        result[self.key_name] = scales

    def _backward_params(self, result):
        self._init_params(result)
        params = result.pop(self.key_name, None)
        if params:
            self.params = tuple(1 / np.array(params))

    def apply_to_img(self, result):
        _transform = Resize(factor=self.params)
        _transform._forward_params(result)
        _transform.apply_to_img(result)
        return result

    def apply_to_seg(self, result):
        _transform = Resize(factor=self.params)
        _transform._forward_params(result)
        _transform.apply_to_seg(result)
        pass

    def apply_to_det(self, result):
        _transform = Resize(factor=self.params)
        _transform._forward_params(result)
        _transform.apply_to_det(result)
        pass

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(img_scales={})'.format(self.scales)
        return repr_str


class MultiGamma(AugBase):
    def __init__(self, gammas: list):
        super().__init__()
        self.always = True
        assert min(gammas) > -1.0 and max(gammas) < 1.0
        self.gammas = gammas.copy()
        self.tmp_gammas = gammas.copy()

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(p={}, gammas={})'.format(self.p, self.gammas)
        return repr_str

    @property
    def canBackward(self):
        return True

    @property
    def repeats(self):
        return len(self.gammas)

    def _forward_params(self, result):
        self._init_params(result)
        if len(self.tmp_gammas) == 0:
            self.tmp_gammas = self.gammas.copy()
        gamma = tuple([self.tmp_gammas.pop(0) + 1] * self.channels)
        self.params = gamma
        result[self.key_name] = self.params

    def _backward_params(self, result):
        self._init_params(result)
        params = result.pop(self.key_name, None)
        if params is not None:
            self.params = tuple([1 / p for p in params])

    def apply_to_img(self, result):
        image = result['img']
        new_image = np.zeros_like(image)
        for c in range(self.channels):
            c_image = image[c]
            temp_min, temp_max = np.min(c_image) - 1e-5, np.max(c_image) + 1e-5
            c_image = (c_image - temp_min) / (temp_max - temp_min)
            c_image = np.power(c_image, self.params[c])
            new_image[c] = c_image * (temp_max - temp_min) + temp_min
        result['img'] = new_image


class Patches(AugBase):
    """
    ONLY USED IN INFERENCE OR EVALUATION
    support segmentation, detection
    support 2D and 3D images
    support forward and backward
    """

    FUSION = {
        'max':  np.maximum,
        'mean': lambda source, target: np.round(np.mean(np.array([source, target]) > 0.5, axis=0)),
        'src': lambda source, target: source
    }

    def __init__(self, patch_size=(128, 128), overlap=0.5, fusion_mode='max'):
        super().__init__()
        self.always = True
        assert fusion_mode in Patches.FUSION.keys()
        self.patch_size = patch_size
        self.overlap = overlap
        self.pad_mode = 'constant'
        self.fusion_mode = fusion_mode
        self.fusion_fun = Patches.FUSION[fusion_mode]
        self._stride = [p - int(p * overlap) for p in patch_size]
        self._base_anchor = [0] * len(patch_size) + list(patch_size)  # zyx

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(patch_size={}, overlap={}, fusion_mode={})'.format(self.patch_size, self.overlap,
                                                                         self.fusion_mode)
        return repr_str

    @property
    def canBackward(self):
        return True

    def _forward_params(self, result):
        self._init_params(result)
        assert len(self.patch_size) == self.dim
        shape = self.image_shape
        # print(shape)
        grid = [np.arange(max((s - 1) // self._stride[i], 1)) * self._stride[i] for i, s in enumerate(shape)]  # zyx
        # print(grid, self._stride)

        ctr = np.meshgrid(*grid, indexing='ij')
        ctr = np.stack([*ctr] * 2, axis=0)
        axes = list(range(1, self.dim + 1))
        base_anchors = np.expand_dims(self._base_anchor, axis=axes)
        anchors = ctr + base_anchors  # 2 * dim -> [z,y,x,z,y,x], grid_z,grid_y,grid_x

        axes = list(range(1, self.dim + 1)) + [0]
        anchors = np.transpose(anchors, tuple(axes))
        anchors = np.reshape(anchors, [-1, 2 * self.dim])
        self.params = anchors
        result[self.key_name] = self.params
        # print(self.params)
        # print(len(self.params))

    def _backward_params(self, result):
        self._init_params(result)
        params = result.pop(self.key_name, None)
        if params is not None:
            self.params = np.array(params)

    def apply_to_img(self, result):
        if self.isForwarding:
            tmp_image = result.pop('img')
            pad_size = self.params[-1, self.dim:]
            diff = np.maximum(np.array(pad_size) - np.array(self.image_shape), 0)
            diff = tuple(zip(np.zeros_like(diff), np.array(diff)))
            tmp_image = np.pad(tmp_image, ((0, 0),) + diff, mode=self.pad_mode)

            patches = []
            for anchor in self.params:
                slices = tuple(slice(anchor[i], anchor[i + self.dim]) for i in range(self.dim))
                slices = (slice(None),) + slices
                patches.append(tmp_image[slices])

            result['patches_img'] = np.stack(patches, axis=0)
            del patches
            del tmp_image
            gc.collect()
            # slices = tuple([slice(lp, shape - rp) for (lp, rp), shape in zip(diff, self.image_shape)])
            # slices = (slice(None),) + slices
            # result['img'] = result['img'][slices]
        else:
            patches = result.pop('patches_img')
            new_image = - np.ones(self.array_shape) * np.inf
            for p, anchor in enumerate(self.params):
                slices = tuple(slice(anchor[i], anchor[i + self.dim]) for i in range(self.dim))
                slices = (slice(None),) + slices
                target = new_image[slices]
                refined_slices = tuple(slice(0, i) for i in target.shape[1:])
                refined_slices = (slice(None),) + refined_slices
                source = patches[p, ...][refined_slices]
                target = np.where(target == -np.inf, source, target)
                new_image[tuple(slices)] = np.mean(np.array([source, target]), axis=0)

            result['img'] = new_image

    def apply_to_seg(self, result):
        if self.isForwarding:
            for key in result.get('seg_fields', []):
                tmp_image = result.pop(key)
                pad_size = self.params[-1, self.dim:]
                diff = np.maximum(np.array(pad_size) - np.array(self.image_shape), 0)
                diff = tuple(zip(np.zeros_like(diff), np.array(diff)))
                tmp_image = np.pad(tmp_image, ((0, 0),) + diff, mode=self.pad_mode)

                patches = []
                for anchor in self.params:
                    slices = tuple(slice(anchor[i], anchor[i + self.dim]) for i in range(self.dim))
                    slices = (slice(None),) + slices
                    patches.append(tmp_image[slices])

                result['patches_' + key] = np.stack(patches, axis=0)
                del patches
                del tmp_image
                # slices = tuple([slice(lp, shape - rp) for (lp, rp), shape in zip(diff, self.image_shape)])
                # slices = (slice(None),) + slices
                # result[key] = result[key][slices]

        else:
            for key in result.get('seg_fields', []):
                patches = result.pop('patches_' + key)
                new_image = - np.ones(self.array_shape)[[0], ...] * np.inf
                for p, anchor in enumerate(self.params):
                    slices = tuple(slice(anchor[i], anchor[i + self.dim]) for i in range(self.dim))
                    slices = (slice(None),) + slices
                    target = new_image[tuple(slices)]
                    refined_slices = tuple(slice(0, i) for i in target.shape[1:])
                    refined_slices = (slice(None),) + refined_slices
                    source = patches[p, ...][refined_slices]
                    target = np.where(target == -np.inf, source, target)
                    new_image[tuple(slices)] = self.fusion_fun(source, target)

                result[key] = new_image

    def apply_to_det(self, result):
        if self.isForwarding:
            for key in result.get('det_fields', []):
                print('\033[31m{}-Warning: Please use Crop instead!\033[0m'.format(self.__class__.__name__))
                patches_bboxes = []
                for p, anchor in enumerate(self.params):
                    # np array
                    start, end = anchor[:self.dim], anchor[self.dim:]
                    cropped_bboxes = cropBBoxes(self.dim, result[key], start[::-1], end[::-1], dim_iou_thr=0.7)
                    patches_bboxes.append(cropped_bboxes)
                result['patches_' + key] = patches_bboxes

        else:
            for key in result.get('det_fields', []):  # gt_bboxes
                patches_bboxes = result.pop('patches_' + key)
                new_bboxes = []
                for p, anchor in enumerate(self.params):
                    start, end = anchor[:self.dim], anchor[self.dim:]
                    padded_bboxes = padBBoxes(self.dim, patches_bboxes[p], start[::-1], end[::-1])
                    new_bboxes.append(padded_bboxes)

                dets = np.concatenate(new_bboxes, axis=0)
                if len(dets):
                    k, _ = nmsNd_numpy(dets[:, [*range(self.dim * 2), -1]], 0.7 ** self.dim)
                    dets = dets[k]
                    # print(dets)
                result[key] = dets


class Repeat(AugBase):
    def __init__(self, times):
        super().__init__()
        self.always = True
        self.times = times

    @property
    def repeats(self):
        return self.times