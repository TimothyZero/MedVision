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

import numpy as np
import scipy.ndimage as ndi
from scipy.ndimage.morphology import binary_dilation

from .aug_base import Stage
from .utils import getSphere


class Property(Stage):
    def __init__(self):
        super(Property, self).__init__()
        self.dim = None
        self.channels = None
        self.properties = {}

    def __repr__(self):
        return self.name + '()'

    def _forward(self, result: dict):
        self.dim = result['img_dim']
        self.channels = result['img'].shape[0]
        self.apply_to_img(result)
        self.apply_to_cls(result)
        self.apply_to_seg(result)
        self.apply_to_det(result)
        result['properties'] = self.properties
        return result

    def apply_to_img(self, result):
        self.properties['filename'] = result['filename']
        self.properties['img_shape'] = np.array(result['img_shape'])
        self.properties['img_spacing'] = np.array(result['img_spacing'])

    def apply_to_cls(self, result):
        if 'gt_cls' in result['cls_fields']:
            self.properties['cls'] = result['gt_cls']

    def apply_to_seg(self, result):
        if 'gt_seg' in result['seg_fields']:
            try:
                img = result['img']
                gt_seg = np.copy(result['gt_seg'])
                unique, counts = np.unique(gt_seg, return_counts=True)
                self.properties['gt_seg_unique'] = unique
                self.properties['gt_seg_counts'] = counts

                gt_seg = np.repeat(gt_seg, self.channels, axis=0)
                self.properties['per_95'] = list(map(lambda a, b: np.percentile(a[b > 0], 99.5), img, gt_seg > 0))
                self.properties['per_05'] = list(map(lambda a, b: np.percentile(a[b > 0], 00.5), img, gt_seg > 0))

                d_gt_seg = binary_dilation(gt_seg, np.expand_dims(getSphere(self.dim, 3, 3), 0))
                self.properties['d3_per_95'] = list(map(lambda a, b: np.percentile(a[b > 0], 99.5), img, d_gt_seg > 0))
                self.properties['d3_per_05'] = list(map(lambda a, b: np.percentile(a[b > 0], 00.5), img, d_gt_seg > 0))

                d_gt_seg = binary_dilation(gt_seg, np.expand_dims(getSphere(self.dim, 5, 5), 0))
                self.properties['d5_per_95'] = list(map(lambda a, b: np.percentile(a[b > 0], 99.5), img, d_gt_seg > 0))
                self.properties['d5_per_05'] = list(map(lambda a, b: np.percentile(a[b > 0], 00.5), img, d_gt_seg > 0))

                labeled, num_objs = ndi.label(gt_seg)
                objs_pro = []
                for i in range(1, num_objs + 1):
                    objs_pro.append(np.sum(labeled == i))
                self.properties['num_objs'] = num_objs
                self.properties['objs_min'] = np.min(objs_pro)
                self.properties['objs_mean'] = np.mean(objs_pro)
                self.properties['objs_max'] = np.max(objs_pro)
            except Exception as e:
                print(e)

    def apply_to_det(self, result):
        if 'gt_det' in result['det_fields']:
            unique, counts = np.unique(result['gt_det'][:, -2], return_counts=True)
            self.properties['gt_det_unique'] = unique
            self.properties['gt_det_counts'] = counts