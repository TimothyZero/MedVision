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

from typing import Union
import numpy as np


# def getSphere(dim, size, diameter):
#     radius = diameter / 2 - 0.5
#     structure = np.zeros((size,) * dim)
#
#     center = [i / 2 for i in structure.shape]
#     ctr = np.meshgrid(*[np.arange(0.5, size)]*dim, indexing='ij')
#     ctr = np.stack(ctr, axis=0)
#     ctr = np.transpose(ctr, [*range(1, dim + 1), 0])
#
#     distance = np.sum(np.power(np.abs(ctr - center), 2), axis=-1)
#     structure = (distance <= radius ** 2).astype(np.float32)
#     return structure
#
#
def iouNd_numpy(anchors, targets, dim=None):
    """
    :param anchors:  [N, (x1,y1,x2,y2) | (x1,y1,z1,x2,y2,z2)]
    :param targets:  [M, (x1,y1,x2,y2) | (x1,y1,z1,x2,y2,z2)]
    :param dim: dimension of bbox
    :return:   IoU:  [N,M]
    """

    if not dim:
        dim = targets.shape[-1] // 2
        assert dim in (2, 3)

    anchors = anchors[..., :2 * dim]
    targets = targets[..., :2 * dim]

    # expand dim
    anchors = np.expand_dims(anchors, axis=1)  # [N, 1, 2*dim]
    targets = np.expand_dims(targets, axis=0)  # [1, M, 2*dim]

    # overlap on each dim
    overlap = np.minimum(anchors[..., dim:], targets[..., dim:])
    overlap = overlap - np.maximum(anchors[..., :dim], targets[..., :dim])
    overlap = np.maximum(0.0, overlap)  # [N,M,dim]

    # intersection
    intersection = np.prod(overlap, axis=-1).astype(np.float)  # [N,M]

    # areas
    area_a = np.prod(anchors[..., dim:] - anchors[..., :dim], axis=-1).astype(np.float)  # [N,1]
    area_b = np.prod(targets[..., dim:] - targets[..., :dim], axis=-1).astype(np.float)  # [1,M]

    # iou
    iou = intersection / (area_a + area_b - intersection)
    return iou


def nmsNd_numpy(dets: np.ndarray, threshold: float):
    """
    :param dets:  [[x1,y1,x2,y2,score],  |  [[x1,y1,z1,x2,y2,z2,score],
                   [x1,y1,x2,y2,score]]  |   [x1,y1,z1,x2,y2,z2,score]]
    :param threshold: for example 0.5
    :return: the rest ids of dets
    """
    dim = dets.shape[-1] // 2
    assert dim in (2, 3), dets.shape

    scores = dets[:, -1].copy()
    bboxes = dets[:, :-1].copy()
    assert bboxes.shape[-1] == 2 * dim

    area = np.prod(bboxes[:, dim:] - bboxes[:, :dim] + 1, axis=-1)
    # print(area)

    order = scores.argsort()[::-1]

    keep = []
    while order.shape[0] > 0:
        i = order[0]
        keep.append(i)

        overlap = np.minimum(bboxes[i, dim:], bboxes[order[1:]][:, dim:])
        overlap = overlap - np.maximum(bboxes[i, :dim], bboxes[order[1:]][:, :dim]) + 1
        overlap = np.maximum(overlap, 0)
        inter = np.prod(overlap, axis=-1)
        # print(inter)

        union = area[i] + area[order[1:]] - inter
        iou = inter / union
        # print(iou)

        index = np.where(iou <= threshold)[0]
        # print(index)

        # similar to soft nmsNd_cuda
        # weight = np.exp(-(iou * iou) / 0.5)
        # scores[order[1:]] = weight * scores[order[1:]]

        order = order[index + 1]

    dets = np.concatenate((bboxes, scores[:, None]), axis=1)
    keep = np.array(keep)
    return keep, dets


def clipBBoxes(dim: int,
               bboxes: np.ndarray,
               image_shape: list) -> np.ndarray:
    """
    Args:
        dim: dimension of image
        bboxes:
            shape is [N, 2*dim], [N, 2*dim + 1 or 2], xyz order
            sometimes, it will be used to handle boxes with 'class' and 'score'
        image_shape: [d,] h, w; zyx order, reversed order of bboxes coords

    Returns:
        clipped bboxes
    """
    if bboxes.size == 0:
        return bboxes
    dim = len(image_shape)
    assert dim <= bboxes.shape[1] // 2, f"image is {dim}D, but bboxes is {bboxes.shape}"

    bboxes[:, :2 * dim] = np.maximum(bboxes[:, :2 * dim], 0)
    bboxes[:, :2 * dim] = np.minimum(bboxes[:, :2 * dim], np.array(image_shape[::-1] * 2) - 1)
    bboxes = bboxes[np.all(bboxes[:, dim:2 * dim] > bboxes[:, :dim], axis=1)]
    return bboxes


def cropBBoxes(dim: int,
               bboxes: np.ndarray,
               start_coord: Union[list, tuple, np.ndarray],
               end_coord: Union[list, tuple, np.ndarray],
               dim_iou_thr: float = 0) -> np.ndarray:
    """
    Args:
        dim: image dimension
        bboxes: [N, >=2 * dim], xyz order
        start_coord: shape is [dim, ], xyz order, e.g. [0, 0, 0]
        end_coord: shape is [dim, ], xyz order, e.g. [96, 96, 96]
        dim_iou_thr:

    Returns:
        cropped bboxes

    """
    start_coord = np.array(start_coord)
    end_coord = np.array(end_coord)
    patch_shape = (end_coord - start_coord).tolist()[::-1]  # zyx order
    assert start_coord.shape[-1] == end_coord.shape[-1] == dim

    cropped_bboxes = bboxes.copy()
    cropped_bboxes[:, :2 * dim] = bboxes[:, :2 * dim] - np.tile(start_coord, 2)
    cropped_bboxes = clipBBoxes(dim, cropped_bboxes, patch_shape)
    assert all(np.all(cropped_bboxes[:, dim:2 * dim] > cropped_bboxes[:, :dim], axis=1))

    padded_bboxes = padBBoxes(dim, cropped_bboxes, start_coord, end_coord)
    iou = iouNd_numpy(padded_bboxes[:, :2 * dim], bboxes[:, :2 * dim])
    iou_per_cropped_bbox = np.max(iou, axis=1)
    cropped_bboxes = cropped_bboxes[iou_per_cropped_bbox > dim_iou_thr ** dim]
    return cropped_bboxes


def padBBoxes(dim: int,
              bboxes: np.ndarray,
              start_coord: Union[list, tuple, np.ndarray],
              end_coord: Union[list, tuple, np.ndarray]) -> np.ndarray:
    """
    Args:
        dim: image dimension
        bboxes: [N, >=2 * dim], xyz order
        start_coord: shape is [dim, ], xyz order, e.g. [0, 0, 0]
        end_coord: shape is [dim, ], xyz order, e.g. [96, 96, 96]

    Returns:
        cropped bboxes

    """
    start_coord = np.array(start_coord)
    end_coord = np.array(end_coord)
    assert start_coord.shape[-1] == end_coord.shape[-1] == dim

    padded_bboxes = bboxes.copy()
    padded_bboxes[:, :2 * dim] = padded_bboxes[:, :2 * dim] + np.tile(start_coord, 2)
    return padded_bboxes
#
#
# def objs2bboxes(objs: list) -> np.ndarray:
#     """convert found objs to bboxes format
#
#     Args:
#         objs: list of slice ( without convert to int)
#
#     Returns:
#         bboxes: np.array
#
#     """
#     bboxes = []
#     for obj in objs:
#         one_det = [i.start for i in obj][::-1] + [i.stop for i in obj][::-1]
#         bboxes.append(one_det)
#     bboxes = np.array(bboxes)
#     return bboxes
#
#
# def bboxes2objs(bboxes: np.ndarray) -> list:
#     """convert bboxes format to objs format
#
#     Args:
#         bboxes: [N, 2 * dim]
#
#     Returns:
#         objs: list of slice
#
#     """
#     dim = bboxes.shape[-1] // 2
#     assert dim in (2, 3), bboxes.shape
#     assert bboxes.ndim == 2
#     objs = []
#     for one_det in bboxes:
#         obj = [slice(one_det[i], one_det[dim + i]) for i in range(dim)]
#         obj = obj[::-1]
#         objs.append(obj)
#     return objs


if __name__ == "__main__":
    start = [0, 0, 0]  # zyx order
    end = [100, 96, 96]  # zyx order

    bboxes = [
        # x1,y1,z1,x2,y2,z2, class, score
        [22, 54, 35, 177, 199, 164, 1, 1.00],
        [32, 54, 45, 67, 87, 99, 2, 1.00],
        [67, 87, 89, 77, 97, 99, 1, 1.00],
        [122, 154, 135, 177, 199, 164, 3, 1.00],
    ]

    # cropped_bboxes = cropBBoxes(3, np.array(bboxes), start[::-1], end[::-1], dim_iou_thr=0.7)
    # print(cropped_bboxes)

    print(np.array(bboxes))

    # objs = bboxes2objs(np.array(bboxes)[:, :6])
    # print(objs)
    #
    # det = objs2bboxes(objs)
    # print(det)