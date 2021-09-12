import os.path as osp
import numpy as np
import gc
from skimage.measure import label, regionprops

from ..io.imageio import ImageIO
from .base import AugBase


class LoadPrepare(object):
    """
    Not a class inherited from Stage. It's used to prepare the format of result.
    """
    def __init__(self, debug=False):
        self.debug = debug

    def __repr__(self):
        return self.__class__.__name__ + f'(debug={self.debug})'

    def __call__(self, image_path, label_path='', **kwargs):
        result = {
            'filename': osp.basename(image_path),
            'image_path': image_path,
            'label_path': label_path,
            'img_fields': [],
            'cls_fields': [],
            'seg_fields': [],
            'det_fields': [],
            'history': [],
            'time': [],
            '_debug_': self.debug
        }
        result.update(kwargs)
        return [result]
        

class LoadImageFromFile(AugBase):
    def __init__(self, to_float32=True):
        super().__init__()
        self.to_float32 = to_float32

    def __repr__(self):
        return self.__class__.__name__ + '(to_float32={})'.format(self.to_float32)
    
    def _forward(self, result):
        image_path = result['image_path']
        
        image, image_dim, image_spacing, image_origin = ImageIO.loadArray(image_path)
        if self.to_float32:
            image = image.astype(np.float32)
        result['img'] = image
        result['img_dim'] = image_dim
        result['img_shape'] = image.shape
        result['img_spacing'] = image_spacing
        result['img_origin'] = image_origin
        result['ori_shape'] = image.shape
        result['ori_spacing'] = image_spacing
        result['img_fields'].append('img')
        gc.collect()
        return result


class LoadAnnotations(AugBase):
    def __init__(self,
                 with_cls=False,
                 with_seg=False,
                 with_det=False):
        super().__init__()
        self.with_cls = with_cls  # for classification
        self.with_seg = with_seg  # for segmentation
        self.with_det = with_det  # for detection
        assert self.with_cls or with_seg or self.with_det

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(with_cls={}, with_seg={}, with_det={})'.format(self.with_cls, self.with_seg, self.with_det)
        return repr_str
    
    def _forward(self, result):
        if self.with_cls:
            self._load_cls(result)
        elif self.with_seg:
            self._load_seg(result)
        elif self.with_det:
            self._load_det(result)
        return result
    
    @staticmethod
    def _load_cls(result):
        """ class is 1 based number"""
        assert isinstance(result['label_path'], int), 'with label must contain a <int> label'
        result['gt_cls'] = result['label_path']
        result['cls_fields'].append('gt_cls')

    @staticmethod
    def _load_seg(result):
        """ seg is [1, d, h, w] """
        label_path = result['label_path']
        assert osp.exists(label_path), 'label path must exist'
        seg, seg_dim, _, _ = ImageIO.loadArray(label_path)
        # assert result['img_dim'] == seg_dim, f"img is {result['img_dim']}D while label is {seg_dim}D"
        if np.max(seg) > 64:  # it should be a cv image, such as jpg
            # the classes should less than 64
            # if 'ISIC2018' in label_path:
            #     seg = (seg > 127.5).astype(np.float32)
            classes = np.unique(seg)
            assert len(classes) < 64, "there maybe some error ?"
            for tag, val in enumerate(classes):
                if tag == val:
                    continue
                seg[seg == val] = tag

        result['gt_seg'] = seg.astype(np.int32)
        result['seg_shape'] = seg.shape
        result['seg_fields'].append('gt_seg')
        gc.collect()

    @staticmethod
    def _load_det(result):
        """ [n, (x,y,x,y, cls, score) | (x,y,z,x,y,z, cls, score)] """
        dim = result['img_dim']
        pseudo_mask = np.zeros_like(result['img'][[0], ...])

        det = []
        for ann in result['label_path']:
            # ann['bbox']: (x,y,w,h) | (x,y,z,w,h,d)
            # ann['category_id']: int, 1 based
            det.append(ann['bbox'] + [ann['category_id'], 1.00])  # the last one is score
            # draw pseudo mask
            bbox = np.array(ann['bbox']).astype(np.float32)
            bbox[dim:] = bbox[:dim] + bbox[dim:]
            slices = list(map(slice, reversed(np.int32(bbox[:dim])), reversed(np.int32(bbox[dim:]))))
            slices = [slice(None)] + slices
            pseudo_mask[tuple(slices)] = ann['category_id']

        det = np.array(det).astype(np.float32)
        if len(det) != 0:
            # RoIAlign will contains start and stop elements, but width should not contain
            # for example:   0123456  start is 1, and stop is 4, width = 4 - 1 + 1 = 4
            #                _####__
            det[:, dim: 2*dim] = det[:, :dim] + det[:, dim: 2*dim] - 1

        result['gt_det'] = det
        result['det_fields'].append('gt_det')
        result['pseudo_mask'] = pseudo_mask
        result['seg_shape'] = pseudo_mask.shape
        result['seg_fields'].append('pseudo_mask')


class LoadCoordinate(AugBase):
    def __repr__(self):
        repr_str = self.__class__.__name__ + '()'
        return repr_str

    def _forward(self, result):
        img_shape = result['img'].shape[1:]
        zz, yy, xx = np.meshgrid(
            np.linspace(-0.5, 0.5, img_shape[0]),
            np.linspace(-0.5, 0.5, img_shape[1]),
            np.linspace(-0.5, 0.5, img_shape[2]),
            indexing='ij')
        coord = np.stack([zz, yy, xx], 0).astype('float32')

        result['gt_coord'] = coord
        result['seg_fields'].append('gt_coord')
        return result


class LoadPseudoAsSeg(AugBase):
    def __repr__(self):
        repr_str = self.__class__.__name__ + '()'
        return repr_str

    def _forward(self, result):
        result['gt_seg'] = result.pop('pseudo_mask')
        result['seg_fields'].remove('pseudo_mask')
        result['seg_fields'].append('gt_seg')
        return result


class LoadSegAsImg(AugBase):
    def __repr__(self):
        repr_str = self.__class__.__name__ + '()'
        return repr_str

    def _forward(self, result):
        result['filename'] = osp.basename(result['label_path'])
        result['image_path'] = result['label_path']
        return result


class LoadWeights(AugBase):
    def __repr__(self):
        repr_str = self.__class__.__name__ + '()'
        return repr_str

    def _forward(self, result):
        dist, _, _, _ = ImageIO.loadArray(result['weight_path'])
        result['pixel_weight'] = dist
        return result


class LoadPredictions(AugBase):
    def __init__(self,
                 with_cls=False,
                 with_seg=False,
                 with_det=False):
        super().__init__()
        self.with_cls = with_cls  # for classification
        self.with_seg = with_seg  # for segmentation
        self.with_det = with_det  # for detection
        assert self.with_cls or with_seg or self.with_det

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(with_cls={}, with_seg={}, with_det={})'.format(self.with_cls, self.with_seg, self.with_det)
        return repr_str

    def _forward(self, result):
        if self.with_cls:
            self._load_cls(result)
        elif self.with_seg:
            self._load_seg(result)
        elif self.with_det:
            self._load_det(result)
        return result

    @staticmethod
    def _load_cls(result):
        result['pred_cls'] = -1
        result['cls_fields'].append('pred_cls')

    @staticmethod
    def _load_seg(result):
        result['pred_seg'] = np.zeros_like(result['img'][[0], ...])
        result['seg_fields'].append('pred_seg')

    @staticmethod
    def _load_det(result):
        dim = result['img_dim']
        result['pred_det'] = np.ones([500, 2 * dim + 2]) * -1.
        result['det_fields'].append('pred_det')


class WorldVoxelConversion(AugBase):
    def __init__(self, reverse=False):
        super(WorldVoxelConversion, self).__init__()
        self.reverse = reverse

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(reverse={})'.format(self.reverse)
        return repr_str

    @staticmethod
    def worldToVoxelCoord(worldCoord, origin, spacing):
        stretchedVoxelCoord = np.absolute(worldCoord - origin)
        voxelCoord = stretchedVoxelCoord / spacing
        return voxelCoord

    @staticmethod
    def VoxelToWorldCoord(voxelCoord, origin, spacing):
        stretchedVoxelCoord = voxelCoord * spacing
        worldCoord = stretchedVoxelCoord + origin
        return worldCoord

    def convert(self, result):
        np.concatenate(
            self.VoxelToWorldCoord(result['gt_det'][:, :3], result['img_spacing'], result['origin']),
            self.VoxelToWorldCoord(result['gt_det'][:, 3:], result['img_spacing'], result['origin']))

    def _forward(self, result: dict):
        pass
        # if self.reverse:
        #     result['gt_det'] =


class Instance2BBoxConversion(AugBase):
    def __init__(self):
        super(Instance2BBoxConversion, self).__init__()
        self.always = True
        self.reverse = False

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(reverse={})'.format(self.reverse)
        return repr_str

    def _forward(self, result: dict):
        assert 'gt_seg' in result['seg_fields']
        self._init_params(result)

        foreground = result['gt_seg'][0].astype(int)
        labeled, num_obj = label(foreground, return_num=True)
        regions = regionprops(labeled, intensity_image=foreground)

        gt_det = []
        for region in regions:
            det = [i.start for i in region.slice][::-1] + [i.stop for i in region.slice][::-1]
            assert int(region.mean_intensity) == region.mean_intensity
            det += [int(region.mean_intensity), 1.0]
            gt_det.append(det)
        result['gt_det'] = np.array(gt_det)

        return result

