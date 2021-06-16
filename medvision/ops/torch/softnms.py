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


def softnmsNd_pytorch(dets: torch.Tensor, threshold: float, method=1, sigma=0.5):
    """
    :param dets:  [[x1,y1,x2,y2,score],  |  [[x1,y1,z1,x2,y2,z2,score],
                   [x1,y1,x2,y2,score]]  |   [x1,y1,z1,x2,y2,z2,score]]
    :param threshold: for example 0.5
    :param method
    :param sigma: gaussian filter sigma
    :return: the index of the selected boxes
    """
    dim = dets.shape[-1] // 2
    assert dim in (2, 3)

    scores = dets[:, -1].clone()
    bboxes = dets[:, :-1].clone()
    assert bboxes.shape[-1] == 2 * dim

    N = bboxes.shape[0]
    if bboxes.is_cuda:
        indexes = torch.arange(0, N, dtype=torch.float).cuda().view(N, 1)
    else:
        indexes = torch.arange(0, N, dtype=torch.float).view(N, 1)
    # Indexes concatenate boxes with the last column
    bboxes = torch.cat((bboxes, indexes), dim=1)

    area = torch.prod(bboxes[:, dim:-1] - bboxes[:, :dim] + 1, dim=-1).float()

    for i in range(N):
        # intermediate parameters for later parameters exchange
        tmp_score = scores[i].clone()
        pos = i + 1

        if i != N - 1:
            max_score, max_pos = torch.max(scores[pos:], dim=0)
            if tmp_score < max_score:
                bboxes[i], bboxes[max_pos.item() + i + 1] = bboxes[max_pos.item() + i + 1].clone(), bboxes[i].clone()
                scores[i], scores[max_pos.item() + i + 1] = scores[max_pos.item() + i + 1].clone(), scores[i].clone()
                area[i], area[max_pos + i + 1] = area[max_pos + i + 1].clone(), area[i].clone()

        overlap = torch.min(bboxes[i, dim:-1], bboxes[pos:, dim:-1])
        overlap = overlap - torch.max(bboxes[i, :dim], bboxes[pos:, :dim]) + 1
        overlap = torch.clamp(overlap, min=0)
        inter = torch.prod(overlap, dim=-1).float()

        union = area[i] + area[pos:] - inter
        iou = inter / union

        if method == 1:
            # linear
            weight = torch.where(iou > threshold, 1 - iou, torch.ones_like(iou))
        else:
            # Gaussian decay
            weight = torch.exp(-(iou * iou) / sigma)
        scores[pos:] = weight * scores[pos:]

    keep = bboxes[:, -1].clone().long()
    bboxes[:, -1] = scores
    return keep, bboxes
