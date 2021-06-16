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

import torch


def nmsNd_pytorch(dets: torch.Tensor, threshold: float):
    """
    :param dets:  [[x1,y1,x2,y2,score],  |  [[x1,y1,z1,x2,y2,z2,score],
                   [x1,y1,x2,y2,score]]  |   [x1,y1,z1,x2,y2,z2,score]]
    :param threshold: for example 0.5
    :return: the rest ids of dets
    """
    dim = dets.shape[-1] // 2
    assert dim in (2, 3), dets.shape

    scores = dets[:, -1].clone()
    bboxes = dets[:, :-1].clone()
    assert bboxes.shape[-1] == 2 * dim, bboxes.shape

    area = torch.prod(bboxes[:, dim:] - bboxes[:, :dim] + 1, dim=-1).float()
    # print(area)

    order = scores.argsort(descending=True)

    keep = []
    while order.shape[0] > 0:
        i = order[0]
        keep.append(i.item())

        overlap = torch.min(bboxes[i, dim:], bboxes[order[1:]][:, dim:])
        overlap = overlap - torch.max(bboxes[i, :dim], bboxes[order[1:]][:, :dim]) + 1
        overlap = torch.clamp(overlap, min=0)
        inter = torch.prod(overlap, dim=-1).float()
        # print(inters)

        union = area[i] + area[order[1:]] - inter
        iou = inter / union
        # print(iou)

        index = torch.where(iou <= threshold)[0]
        # print(index)

        # similar to soft nms_nd
        # weight = torch.exp(-(iou * iou) / 0.5)
        # scores[order[1:]] = weight * scores[order[1:]]

        order = order[index + 1]

    dets = torch.cat((bboxes, scores.unsqueeze(-1)), dim=1)
    keep = torch.tensor(keep)
    return keep, dets


