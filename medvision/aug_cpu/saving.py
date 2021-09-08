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

from copy import deepcopy
import numpy as np
import os
import os.path as osp

from .base import AugBase
from ..io.imageio import ImageIO


# TODO implement with forward and backward

class SaveFolder(AugBase):
    def __init__(self, folder):
        super().__init__()
        self.folder = folder
        os.makedirs(self.folder, exist_ok=True)

    def __repr__(self):
        return self.__class__.__name__ + '(folder="{}")'.format(self.folder)

    def _forward(self, result: dict):
        result['SaveFolder'] = self.folder
        return result


class SaveImageToFile(AugBase):
    def __init__(self, ext='same'):
        super().__init__()
        self.ext = ext if ext.startswith('.') else '.' + ext

    def __repr__(self):
        return self.__class__.__name__ + '(ext="{}")'.format(self.ext)

    def _forward(self, result: dict):
        _ext = osp.splitext(osp.basename(result['image_path']))[-1]
        if _ext == '.gz':
            _ext = '.nii.gz'
        if not self.ext == '.same':
            _ext = self.ext
        result_path = osp.join(result['image_path'] + "_img" + _ext)
        if result.get('SaveFolder', None):
            result_path = osp.join(result['SaveFolder'], osp.basename(result_path))
        image = result['img']
        ImageIO.saveArray(result_path, image, result['img_spacing'], result['img_origin'])
        return result


class SaveAnnotations(AugBase):
    def __init__(self,
                 with_det=False,
                 with_cls=False,
                 with_seg=False):
        super().__init__()
        self.with_det = with_det  # for detection
        self.with_cls = with_cls  # for classification
        self.with_seg = with_seg  # for segmentation
        assert self.with_det or self.with_cls or with_seg

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(with_det={}, with_cls={}, with_seg={})'.format(
            self.with_det, self.with_cls, self.with_seg)
        return repr_str

    def _forward(self, result):
        if self.with_seg:
            self._save_seg(result)
        elif self.with_cls:
            self._save_cls(result)
        elif self.with_det:
            self._save_det(result)
        return result

    @staticmethod
    def _save_cls(result):
        _ext = osp.splitext(osp.basename(result['image_path']))[-1]
        if _ext == '.gz':
            _ext = '.nii.gz'
        _dir = osp.dirname(result['image_path'])
        pred_result = result['pred_cls']
        result_path = osp.join(result['image_path'] + "_cls")
        if result.get('SaveFolder', None):
            result_path = osp.join(result['SaveFolder'], osp.basename(result_path))
        np.save(result_path, pred_result)

    @staticmethod
    def _save_seg(result):
        if not result['label_path']:
            _ext = '.nii.gz' if result['img_dim'] == 3 else '.png'
            t = list(os.path.split(result['image_path']))
            t[-1] = t[-1] + '_seg' + _ext
            result_path = '/'.join(t)
        else:
            _ext = osp.splitext(osp.basename(result['label_path']))[-1]
            if _ext == '.gz':
                _ext = '.nii.gz'
            result_path = osp.join(result['label_path'] + "_seg" + _ext)
        if result.get('SaveFolder', None):
            result_path = osp.join(result['SaveFolder'], osp.basename(result_path))
        if 'gt_seg' not in result.keys():
            gt_seg = result['pred_seg']
        else:
            gt_seg = result['gt_seg']
        ImageIO.saveArray(result_path, gt_seg, result['img_spacing'], result['img_origin'])

    @staticmethod
    def _save_det(result):
        dim = result['img_dim']
        _dir = osp.dirname(result['image_path'])

        pred_result = result['pred_det']
        result_path = osp.join(_dir, osp.basename(result['image_path']) + "_det")
        if result.get('SaveFolder', None):
            result_path = osp.join(result['SaveFolder'], osp.basename(result_path))
        np.save(result_path, pred_result)

        pseudo_mask = np.zeros_like(result['img'][[0], ...])

        for ann in pred_result:
            bbox = ann[: 2 * dim]
            slices = list(map(slice, reversed(np.int32(bbox[:dim])), reversed(np.int32(bbox[dim:]))))
            slices = [slice(None)] + slices
            pseudo_mask[tuple(slices)] = ann[-2]
        if dim == 3:
            _ext = '.nii.gz'
        else:
            _ext = '.png'
        ImageIO.saveArray(result_path + _ext, pseudo_mask, result['img_spacing'], result['img_origin'])


class SplitPatches(AugBase):
    def _forward(self, result: dict):
        assert 'patches_img' in result.keys()

        patched_keys = []
        for k in result.keys():
            if k.startswith('patches_'):
                patched_keys.append(k.replace('patches_', ''))

        results = []
        for p in range(len(result['patches_img'])):
            # print(p)
            r = {
                'filename':    result['filename'],
                'img_dim':     result['img_dim'],
                'img_spacing': result['img_spacing'],
                'img_origin':  result['img_origin'],
                'history':     deepcopy(result['history']),
                'time':        deepcopy(result['time']),
                '_debug_':     result['_debug_']
            }

            if result['image_path'] is not None:
                fs = list(os.path.splitext(result['image_path']))
                fs.insert(1, f"_{p:04d}")
                r['image_path'] = "".join(fs)

            if result['label_path'] is not None:
                fs = list(os.path.splitext(result['label_path']))
                fs.insert(1, f"_{p:04d}")
                r['label_path'] = "".join(fs)

            for k in patched_keys:
                r[k] = result['patches_' + k][p]
            results.append(r)
        return results

    def forward(self, result: dict):
        return [self._post_forward(r) for r in self._forward(self._pre_forward(result))]

    def multi_forward(self, result_list: list):
        return [r for result in result_list for r in self.forward(result)]
