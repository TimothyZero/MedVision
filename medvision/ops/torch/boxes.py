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


def iouNd_pytorch(anchors: torch.Tensor,
                  targets: torch.Tensor,
                  dim=None,
                  mode='iou',
                  is_aligned=False, 
                  eps=1e-6) -> torch.Tensor:
    """
    :param anchors:  [N, (y1,x1,y2,x2) | (y1,x1,z1,y2,x2,z2)]
    :param targets:  [M, (y1,x1,y2,x2) | (y1,x1,z1,y2,x2,z2)]
    :param dim: dimension of bbox
    :param mode:
    :param is_aligned: if N == M
    :param eps:
    :return:   IoU:  [N,M]
    """
    assert mode in ['iou', 'iof', 'giou', 'diou'], f'Unsupported mode {mode}'
    
    if not dim:
        dim = targets.shape[-1] // 2
        assert dim in (2, 3)

    if is_aligned:
        assert anchors.shape[:-1] == targets.shape[:-1]

    anchors = anchors[..., :2 * dim]
    targets = targets[..., :2 * dim]

    if not is_aligned:
        # expand dim
        anchors = torch.unsqueeze(anchors, dim=1)  # [N, 1, 2*dim]
        targets = torch.unsqueeze(targets, dim=0)  # [1, M, 2*dim]

    # overlap on each dim
    lt = torch.max(anchors[..., :dim], targets[..., :dim])  # [N, M, dim]
    rb = torch.min(anchors[..., dim:], targets[..., dim:])  # [N, M, dim]
    wh = rb - lt
    wh = torch.max(torch.zeros_like(wh), wh)

    # intersection
    intersection = torch.prod(wh, dim=-1).float()  # [N,M]

    # areas
    area_a = torch.prod(anchors[..., dim:] - anchors[..., :dim], dim=-1).float()  # [N,1]
    area_b = torch.prod(targets[..., dim:] - targets[..., :dim], dim=-1).float()  # [1,M]

    # union
    if mode == 'iof':
        union = area_b
    else:
        union = area_a + area_b - intersection  # [N, M]

    # iou
    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    iou = intersection / union  # [N, M]
    
    if mode in ['iou', 'iof']:
        return iou
    elif mode == 'giou':
        enclosed_lt = torch.min(anchors[..., :dim], targets[..., :dim])  # [N, M, dim]
        enclosed_rb = torch.max(anchors[..., dim:], targets[..., dim:])  # [N, M, dim]
        enclosed_wh = enclosed_rb - enclosed_lt
        enclosed_area = torch.prod(enclosed_wh, dim=-1).float()  # [N,M]
        enclosed_area = torch.max(enclosed_area, eps)

        giou = iou - (enclosed_area - union) / enclosed_area
        return giou
    elif mode == 'diou':
        enclosed_lt = torch.min(anchors[..., :dim], targets[..., :dim])  # [N, M, dim]
        enclosed_rb = torch.max(anchors[..., dim:], targets[..., dim:])  # [N, M, dim]
        enclosed_wh = enclosed_rb - enclosed_lt
        c2 = torch.sum(torch.pow(enclosed_wh, 2), dim=-1) + eps  # [N,M]
        
        center_a = (anchors[..., :dim] + anchors[..., dim:]) / 2  # [N, M, dim]
        center_t = (targets[..., :dim] + targets[..., dim:]) / 2  # [N, M, dim]
        center_wh = center_a - center_t
        d2 = torch.sum(torch.pow(center_wh, 2), dim=-1) + eps

        diou = iou - d2 / c2
        return diou
    

def distNd_pytorch(anchors: torch.Tensor,
                   targets: torch.Tensor,
                   dim=None) -> torch.Tensor:
    """
    :param anchors:  [N, (y1,x1,y2,x2) | (y1,x1,z1,y2,x2,z2)]
    :param targets:  [M, (y1,x1,y2,x2) | (y1,x1,z1,y2,x2,z2)]
    :param dim: dimension of bbox
    :return:   IoU:  [N,M]
    """
    if not dim:
        dim = targets.shape[-1] // 2
        assert dim in (2, 3)

    anchors = anchors[..., :2 * dim]
    targets = targets[..., :2 * dim]

    # expand dim
    anchors = torch.unsqueeze(anchors, dim=1)  # [N, 1, 2*dim]
    targets = torch.unsqueeze(targets, dim=0)  # [1, M, 2*dim]

    # center on each dim
    anchors = (anchors[..., :dim] + anchors[..., dim:]) / 2  # [N, 1, dim]
    targets = (targets[..., :dim] + targets[..., dim:]) / 2  # [1, M, dim]

    # distance
    distance = torch.pow(anchors - targets, 2).float()  # [N, M, dim]
    distance = torch.sqrt(torch.sum(distance, dim=-1))

    return distance


def deltaNd_pytorch(anchors: torch.Tensor,
                    targets: torch.Tensor,
                    means: torch.Tensor = None,
                    stds: torch.Tensor = None):
    """
    :param  anchors: [N, (y1,x1,y2,x2) | (y1,x1,z1,y2,x2,z2)]
    :param  targets: [N, (y1,x1,y2,x2) | (y1,x1,z1,y2,x2,z2)]
    :param  means:
    :param  stds:
    :return: regression: [N, (dy,dx,dh,dw) | (dy,dx,dz,dh,dw,dd)]
    """

    dim = anchors.shape[-1] // 2
    assert dim in (2, 3)
    assert len(anchors) == len(targets)
    if means is None:
        means = torch.tensor([[0.0] * 2 * dim]).to(anchors.device)
    if stds is None:
        stds = torch.tensor([[0.1] * dim + [0.2] * dim]).to(anchors.device)
    means = means.to(anchors.device)
    stds = stds.to(anchors.device)

    # shape of anchor and target
    anchor_shape = anchors[..., dim:] - anchors[..., :dim]
    target_shape = targets[..., dim:] - targets[..., :dim]
    # print("anchor_shape\n", anchor_shape)
    # print("target_shape\n", target_shape)

    # center of anchor and target
    anchor_center = 0.5 * (anchors[..., dim:] + anchors[..., :dim])
    target_center = 0.5 * (targets[..., dim:] + targets[..., :dim])
    # print("anchor_center\n", anchor_center)
    # print("target_center\n", target_center)

    # delta of shape and center
    delta_center = (target_center - anchor_center) / anchor_shape
    delta_shape = torch.log(target_shape / anchor_shape)
    # delta_shape = 1 - torch.exp(1 - (anchor_shape / target_shape))
    # print("delta_center\n", delta_center)
    # print("delta_shape\n", delta_shape)

    deltas = torch.cat([delta_center, delta_shape], dim=1)
    # print(regression.shape)

    # target /= np.array([0.1, 0.1, 0.2, 0.2])
    deltas = (deltas - means) / stds
    return deltas


def applyDeltaNd_pytorch(anchors: torch.Tensor,
                         deltas: torch.Tensor,
                         means: torch.Tensor = None,
                         stds: torch.Tensor = None):
    """
    make sure coords is last dim
    应用回归目标到边框,用rpn网络预测的delta refine anchor
    :param anchors: [N, (y1, x1, y2, x2)]
    :param deltas:  [N, (dy,dx,dh,dw)]
    :param  means:
    :param  stds:
    :return:        [N, (y1, x1, y2, x2)]
    """
    dim = anchors.shape[-1] // 2
    assert dim in (2, 3)
    assert len(anchors) == len(deltas), f"deltas: {deltas.shape} anchors: {anchors.shape}"
    if means is None:
        means = torch.tensor([[0.0] * 2 * dim])
    if stds is None:
        stds = torch.tensor([[0.1] * dim + [0.2] * dim])
    means = means.to(anchors.device)
    stds = stds.to(anchors.device)

    deltas = deltas * stds + means

    anchor_shape = anchors[..., dim:] - anchors[..., :dim]
    # print(anchor_shape)

    anchor_center = 0.5 * (anchors[..., dim:] + anchors[..., :dim])
    # print(anchor_center)

    anchor_center = anchor_center + deltas[..., :dim] * anchor_shape
    # print(anchor_center)

    anchor_shape = anchor_shape * torch.exp(deltas[..., dim:])
    # anchor_shape = anchor_shape / (1 - torch.log(1 - deltas[..., dim:]))
    # print(anchor_shape)

    anchors_refined = torch.cat([anchor_center - 0.5 * anchor_shape,
                                 anchor_center + 0.5 * anchor_shape], dim=-1)

    return anchors_refined


def bbox2roi(bbox_list):
    """Convert a list of bboxes to roi format.

    Args:
        bbox_list (list[Tensor]): a list of bboxes corresponding to a batch
            of images.

    Returns:
        Tensor: shape (n, 1 + 2 dim), [batch_ind, x1, y1, z1, x2, y2, z2]
    """
    rois_list = []
    for img_id, bboxes in enumerate(bbox_list):
        if bboxes.size(0) > 0:
            img_inds = bboxes.new_full((bboxes.size(0), 1), img_id)
            rois = torch.cat([img_inds, bboxes], dim=-1)
        else:
            rois = bboxes.new_zeros((0, 1 + bboxes.shape[1]))
        rois_list.append(rois)
    rois = torch.cat(rois_list, 0)
    return rois


def enlarge(rois: torch.Tensor, scale_factor: float):
    dim = rois.shape[-1] // 2
    shape = rois[:, dim:] - rois[:, :dim]
    enlarged = rois.clone()
    enlarged[:, dim:] = enlarged[:, dim:] + (scale_factor - 1) * shape / 2
    enlarged[:, :dim] = enlarged[:, :dim] - (scale_factor - 1) * shape / 2
    return [enlarged]
