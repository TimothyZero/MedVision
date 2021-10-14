# -*- coding:utf-8 -*-
import torch
import numpy as np

from .base import CudaAugBase
from .utils import cropBBoxes, padBBoxes, nmsNd_numpy
from ..ops.cuda_fun_tools import affine_2d, affine_3d


class CudaPatches(CudaAugBase):
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
        assert fusion_mode in CudaPatches.FUSION.keys()
        self.always = True
        self.patch_size = patch_size
        self.overlap = overlap
        self.fusion_mode = fusion_mode
        self.fusion_fun = CudaPatches.FUSION[fusion_mode]
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
        if self.dim == 3:
            anchors = anchors[:, [2, 1, 0, 5, 4, 3]]
        else:
            anchors = anchors[:, [1, 0, 3, 2]]
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
            for key in result.get('img_fields', []):
                image = result[key]
                device = image.device

                if self.dim == 2:
                    cuda_fun = affine_2d
                else:
                    cuda_fun = affine_3d

                index = torch.FloatTensor([[0]] * len(self.params))
                bboxes = torch.FloatTensor(self.params)  # xyz order, shape is [n, 2*dim]
                center = bboxes[:, :self.dim] / 2 + bboxes[:, self.dim:] / 2
                shape = bboxes[:, self.dim:] - bboxes[:, :self.dim]
                if self.dim == 2:
                    angles = torch.FloatTensor([[0]] * len(self.params))
                else:
                    angles = torch.FloatTensor([[0, 0, 0]] * len(self.params))

                rois = torch.cat([index, center, shape, angles], dim=1).to(device)
                out_size = self.patch_size
                spatial_scale = 1
                aligned = True
                order = 1

                patches = cuda_fun(
                    image.unsqueeze(0),
                    rois,
                    out_size,
                    spatial_scale,
                    1,
                    aligned,
                    order
                ).squeeze(0)
                result[f'patches_{key}'] = patches

    def apply_to_seg(self, result):
        if self.isForwarding:
            for key in result.get('seg_fields', []):
                image = result[key]
                device = image.device

                if self.dim == 2:
                    cuda_fun = affine_2d
                else:
                    cuda_fun = affine_3d

                index = torch.FloatTensor([[0]] * len(self.params))
                bboxes = torch.FloatTensor(self.params)  # xyz order, shape is [n, 2*dim]
                center = bboxes[:, :self.dim] / 2 + bboxes[:, self.dim:] / 2
                shape = bboxes[:, self.dim:] - bboxes[:, :self.dim]
                if self.dim == 2:
                    angles = torch.FloatTensor([[0]] * len(self.params))
                else:
                    angles = torch.FloatTensor([[0, 0, 0]] * len(self.params))

                rois = torch.cat([index, center, shape, angles], dim=1).to(device)
                out_size = self.patch_size
                spatial_scale = 1
                aligned = True
                order = 0

                patches = cuda_fun(
                    image.unsqueeze(0),
                    rois,
                    out_size,
                    spatial_scale,
                    1,
                    aligned,
                    order
                ).squeeze(0)
                result[f'patches_{key}'] = patches

        # else:
        #     for key in result.get('seg_fields', []):
        #         patches = result.pop('patches_' + key)
        #         new_image = - np.ones(self.array_shape)[[0], ...] * np.inf
        #         for p, anchor in enumerate(self.params):
        #             slices = tuple(slice(anchor[i], anchor[i + self.dim]) for i in range(self.dim))
        #             slices = (slice(None),) + slices
        #             target = new_image[tuple(slices)]
        #             refined_slices = tuple(slice(0, i) for i in target.shape[1:])
        #             refined_slices = (slice(None),) + refined_slices
        #             source = patches[p, ...][refined_slices]
        #             target = np.where(target == -np.inf, source, target)
        #             new_image[tuple(slices)] = self.fusion_fun(source, target)
        #
        #         result[key] = new_image

    def apply_to_det(self, result):
        if self.isForwarding:
            for key in result.get('det_fields', []):
                print('\033[31m{}-Warning: Please use Crop instead!\033[0m'.format(self.__class__.__name__))
                bboxes = result[key].cpu().numpy()
                patches_bboxes = []
                for p, anchor in enumerate(self.params):
                    # np array
                    start, end = anchor[:self.dim], anchor[self.dim:]
                    cropped_bboxes = cropBBoxes(self.dim, bboxes, start[::-1], end[::-1], dim_iou_thr=0.7)
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
                result[f'patches_{key}'] = dets
