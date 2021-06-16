# -*- coding:utf-8 -*-
from typing import Union, List

import torch
from torch import nn
from torch.nn import functional as F


class RoIAlign(nn.Module):
    def __init__(self, output_size, spatial_scale=1., sampling_ratio=-1):
        super(RoIAlign, self).__init__()
        assert isinstance(output_size, (list, tuple)), 'output size should be list or tuple'
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio  # not used

        self.dim = len(output_size)

        self.aligned = False  # align_corners
        self.max_parallel_in_sampling = 16

    def forward(self,
                feat_map: torch.Tensor,
                bboxes: Union[List[torch.Tensor], torch.Tensor]):
        """

        :param feat_map: Tensor (N, C, D, H, W)
        :param bboxes: list of Tensors [(K_i, 6), ...]
            for K is roi count and 6 stands for x1, y1, z1, x2, y2, z2
            or Tensor [S_K, 1 + 6/ 4]
        :return:
            boxes are Tensor: Tensor (N, K, C, D, H, W)
            boxes are list: list of Tensors [(K_i, C, D, H, W), ...]
        """
        if isinstance(bboxes, list):
            result_list = list()
            for batch_feat, batch_rois in zip(feat_map, bboxes):
                k = batch_rois.shape[0]

                feat = batch_feat.expand(k, *batch_feat.shape).to(feat_map.device)
                grids = self.generate_grids(batch_rois, feat.shape).to(feat_map.device)

                output_feat = torch.cat([F.grid_sample(feat, sub_grids, mode='bilinear', align_corners=self.aligned)
                                         for sub_grids in torch.split(grids, self.max_parallel_in_sampling)])
                result_list.append(output_feat)
            return torch.cat(result_list, dim=0)
        else:
            result_list = list()
            for roi in bboxes:
                batch_idx, bbox = int(roi[0]), roi[1:]
                feat = feat_map[batch_idx].unsqueeze(0)
                grids = self.generate_grids(bbox.unsqueeze(0), feat.shape).to(feat_map.device)

                output_feat = torch.cat([F.grid_sample(feat, sub_grids, mode='bilinear', align_corners=self.aligned)
                                         for sub_grids in torch.split(grids, self.max_parallel_in_sampling)])
                result_list.append(output_feat)
            return torch.cat(result_list, dim=0)

    def generate_grids(self, batch_rois, feat_shape):
        """
        Align corners = True
        A @ (-1) + b = 2 * x1 / s - 1
        A @ 1 + b = 2 * x2 / s - 1
        A = diag((x2 - x1) / s)
        b = (x2 + x1) / s - 1

        Align corners = False
        A @ (-1 + 1 / s) + b = 2 * x1 / s - 1 + 1 / s
        A @ (1 - 1 / s) + b = 2 * x2 / s - 1 + 1 / s
        A = diag((x2 - x1) / (s - 1))
        b = (x2 + x1 + 1) / s - 1

        :param batch_rois: (K, 6)
        :param feat_shape: K, C, D, H, W
        :return:
        """

        k, c = feat_shape[0], feat_shape[1]
        feat_size = torch.tensor(feat_shape[2:][::-1], dtype=torch.float32).to(batch_rois.device)

        x1 = batch_rois[:, :self.dim] / self.spatial_scale
        x2 = batch_rois[:, self.dim:] / self.spatial_scale

        if self.aligned:
            scaling = torch.diag_embed((x2 - x1) / feat_size)
            translation = ((x2 + x1) / feat_size - 1).unsqueeze(2)
        else:
            scaling = torch.diag_embed((x2 - x1) / (feat_size - 1))
            translation = ((x2 + x1 + 1) / feat_size - 1).unsqueeze(2)

        theta = torch.cat([scaling, translation], dim=-1)
        grids = torch.affine_grid_generator(theta, size=[k, c, *self.output_size], align_corners=self.aligned)
        return grids

    def __repr__(self) -> str:
        tmpstr = self.__class__.__name__ + '('
        tmpstr += 'output_size=' + str(self.output_size)
        tmpstr += ', spatial_scale=' + str(self.spatial_scale)
        tmpstr += ', sampling_ratio=' + str(self.sampling_ratio)
        tmpstr += ', aligned=' + str(self.aligned)
        tmpstr += ')'
        return tmpstr