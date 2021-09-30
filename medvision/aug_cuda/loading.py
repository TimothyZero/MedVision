import os.path as osp
import numpy as np
import gc
from skimage.measure import label, regionprops

import torch

from ..io.imageio import ImageIO
from .base import CudaAugBase


class CudaLoadPrepare(object):
    """
    Not a class inherited from Stage. It's used to prepare the format of result.
    """
    def __init__(self, debug=False):
        self.debug = debug

    def __repr__(self):
        return self.__class__.__name__ + f'(debug={self.debug})'

    def __call__(self, image_path, **kwargs):
        result = {
            'filename': osp.basename(image_path),
            'image_path': image_path,
            'img_fields': [],
            'cls_fields': [],
            'seg_fields': [],
            'det_fields': [],
            'history': [],
            'time': [],
            'memory': [],
            '_debug_': self.debug
        }
        result.update(kwargs)
        return [result]
        

class CudaLoadImageFromFile(CudaAugBase):
    def __init__(self, to_float16=False):
        super().__init__()
        self.always = True
        self.to_float16 = to_float16

    def __repr__(self):
        return self.__class__.__name__ + '(to_float16={})'.format(self.to_float16)
    
    def _forward(self, result):
        image_path = result['image_path']
        
        image, image_dim, image_spacing, image_origin = ImageIO.loadArray(image_path)
        if self.to_float16:
            image = image.astype(np.float16)
        else:
            image = image.astype(np.float32)
        result['img'] = torch.from_numpy(image).cuda()
        result['img_dim'] = image_dim
        result['img_shape'] = image.shape
        result['img_spacing'] = image_spacing
        result['img_origin'] = image_origin
        result['ori_shape'] = image.shape
        result['ori_spacing'] = image_spacing
        result['img_fields'].append('img')
        del image
        gc.collect()
        return result


class CudaLoadAnnotations(CudaAugBase):
    def __init__(self,
                 with_cls=False,
                 with_seg=False,
                 with_det=False):
        super().__init__()
        self.always = True
        self.with_cls = with_cls  # for classification
        self.with_seg = with_seg  # for segmentation
        self.with_det = with_det  # for detection
        # assert self.with_cls or with_seg or self.with_det

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(with_cls={}, with_seg={}, with_det={})'.format(self.with_cls, self.with_seg, self.with_det)
        return repr_str
    
    def _forward(self, result):
        if self.with_cls:
            self._load_cls(result)
        if self.with_seg:
            self._load_seg(result)
        if self.with_det:
            self._load_det(result)
        return result
    
    @staticmethod
    def _load_cls(result):
        """ class is 1 based number"""
        assert isinstance(result['cls'], int), 'with label must contain a <int> label'
        result['gt_cls'] = result['cls']
        result['cls_fields'].append('gt_cls')

    @staticmethod
    def _load_seg(result):
        """ seg is [1, d, h, w] """
        label_path = result['seg']
        assert osp.exists(label_path), 'label path must exist'
        seg, seg_dim, _, _ = ImageIO.loadArray(label_path)
        # assert result['img_dim'] == seg_dim, f"img is {result['img_dim']}D while label is {seg_dim}D"
        if np.max(seg) > 64 and seg_dim == 2:  # it should be a cv image, such as jpg
            # the classes should less than 64
            # if 'ISIC2018' in label_path:
            #     seg = (seg > 127.5).astype(np.float32)
            classes = np.unique(seg)
            assert len(classes) < 64, "there maybe some error ?"
            for tag, val in enumerate(classes):
                if tag == val:
                    continue
                seg[seg == val] = tag

        result['gt_seg'] = torch.from_numpy(seg.astype(np.int32)).cuda()
        result['seg_shape'] = seg.shape
        result['seg_fields'].append('gt_seg')
        del seg
        gc.collect()

    @staticmethod
    def _load_det(result):
        """ [n, (x,y,x,y, cls, score) | (x,y,z,x,y,z, cls, score)] """
        dim = result['img_dim']
        # pseudo_mask = np.zeros_like(result['img'][[0], ...])

        det = []
        for ann in result['det']:
            # ann['bbox']: (x,y,w,h) | (x,y,z,w,h,d)
            # ann['category_id']: int, 1 based
            det.append(ann['bbox'] + [ann['category_id'], 1.00])  # the last one is score
            # draw pseudo mask
            # bbox = np.array(ann['bbox']).astype(np.float32)
            # bbox[dim:] = bbox[:dim] + bbox[dim:]
            # slices = list(map(slice, reversed(np.int32(bbox[:dim])), reversed(np.int32(bbox[dim:]))))
            # slices = [slice(None)] + slices
            # pseudo_mask[tuple(slices)] = ann['category_id']

        det = np.array(det).astype(np.float32)
        if len(det) != 0:
            # RoIAlign will contains start and stop elements, but width should not contain
            # for example:   0123456  start is 1, and stop is 4, width = 4 - 1 + 1 = 4
            #                _####__
            det[:, dim: 2*dim] = det[:, :dim] + det[:, dim: 2*dim] - 1

        result['gt_det'] = det
        result['det_fields'].append('gt_det')
        # result['pseudo_mask'] = pseudo_mask
        # result['seg_shape'] = pseudo_mask.shape
        # result['seg_fields'].append('pseudo_mask')


class CudaAnnotationMap(CudaAugBase):
    def __init__(self, mapping: dict, apply_to=('cls', 'det', 'seg')):
        super().__init__()
        self.always = True
        self.mapping = mapping
        self.apply_to = apply_to
        assert isinstance(self.mapping, dict)
        assert all([isinstance(i, int) for i in self.mapping.keys()])
        assert all([isinstance(i, int) for i in self.mapping.values()])

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(mapping={}, apply_to={})'.format(self.mapping, self.apply_to)
        return repr_str

    @property
    def canBackward(self):
        flag = len(set(self.mapping.values())) == len(self.mapping.values())
        flag = flag and len(set(self.mapping.keys())) == len(self.mapping.keys())
        return flag

    def _forward_params(self, result):
        self._init_params(result)
        self.params = self.mapping.copy()
        result[self.key_name] = self.params

    def _backward_params(self, result):
        self._init_params(result)
        params = result.pop(self.key_name, None)
        self.params = dict((v, k) for k, v in params.items())

    def apply_to_cls(self, result):
        if 'cls' in self.apply_to:
            for key in result.get('cls_fields', []):
                for prev, curr in self.params.items():
                    result[key] = curr if result[key] == prev else result[key]
        return result

    def apply_to_seg(self, result):
        if 'seg' in self.apply_to:
            for key in result.get('seg_fields', []):
                for prev, curr in self.params.items():
                    result[key][result[key] == prev] = curr
        return result

    def apply_to_det(self, result):
        if 'det' in self.apply_to:
            for key in result.get('det_fields', []):
                for prev, curr in self.params.items():
                    bboxes_labels = result[key][:, 2 * self.dim]
                    bboxes_labels = np.where(bboxes_labels == prev, curr, bboxes_labels)
                    result[key][:, 2 * self.dim] = bboxes_labels


class CudaInstance2BBoxConversion(CudaAugBase):
    def __init__(self, instance='gt_seg'):
        super(CudaInstance2BBoxConversion, self).__init__()
        self.always = True
        self.instance_key = instance
        self.reverse = False

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(instance_key={}, reverse={})'.format(self.instance_key, self.reverse)
        return repr_str

    def _forward(self, result: dict):
        assert self.instance_key in result['seg_fields']
        self._init_params(result)

        foreground = result[self.instance_key][0].cpu().numpy().astype(int)
        labeled, num_obj = label(foreground, return_num=True)
        regions = regionprops(labeled, intensity_image=foreground)

        gt_det = []
        for region in regions:
            det = [i.start for i in region.slice][::-1] + [i.stop for i in region.slice][::-1]
            assert int(region.mean_intensity) == region.mean_intensity
            det += [int(region.mean_intensity), 1.0]
            gt_det.append(det)
        result['gt_det'] = torch.from_numpy(np.array(gt_det)).cuda()
        result['det_fields'].append('gt_det')
        return result