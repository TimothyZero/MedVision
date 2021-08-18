# -*- coding:utf-8 -*-
import torch
from medvision import _C


def bbox_overlaps_nd(bboxes1, bboxes2, mode='iou', aligned=False, offset=0):
    """Calculate overlap between two set of bboxes.
    modified from https://github.com/open-mmlab/mmdetection

    If ``aligned`` is ``False``, then calculate the ious between each bbox
    of bboxes1 and bboxes2, otherwise the ious between each aligned pair of
    bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (m, 4/6) in <x1, y1, x2, y2> or <x1, y1, z1, x2, y2, z2> format or empty.
        bboxes2 (Tensor): shape (n, 4/6) in <x1, y1, x2, y2> or <x1, y1, z1, x2, y2, z2> format or empty.
            If aligned is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union) or iof (intersection over
            foreground).
        aligned
        offset

    Returns:
        ious(Tensor): shape (m, n) if aligned == False else shape (m, 1)

    Example:
        >>> bboxes1 = torch.FloatTensor([
        >>>     [0, 0, 10, 10],
        >>>     [10, 10, 20, 20],
        >>>     [32, 32, 38, 42],
        >>> ])
        >>> bboxes2 = torch.FloatTensor([
        >>>     [0, 0, 10, 20],
        >>>     [0, 10, 10, 19],
        >>>     [10, 10, 20, 20],
        >>> ])
        >>> bbox_overlaps_nd(bboxes1, bboxes2)
        tensor([[0.5000, 0.0000, 0.0000],
                [0.0000, 0.0000, 1.0000],
                [0.0000, 0.0000, 0.0000]])

    Example:
        >>> empty = torch.FloatTensor([])
        >>> nonempty = torch.FloatTensor([
        >>>     [0, 0, 10, 9],
        >>> ])
        >>> assert tuple(bbox_overlaps_nd(empty, nonempty).shape) == (0, 1)
        >>> assert tuple(bbox_overlaps_nd(nonempty, empty).shape) == (1, 0)
        >>> assert tuple(bbox_overlaps_nd(empty, empty).shape) == (0, 0)
    """

    mode_dict = {'iou': 0, 'iof': 1}
    assert mode in mode_dict.keys()
    mode_flag = mode_dict[mode]
    # Either the boxes are empty or the length of boxes' last dimension is 4
    # assert (bboxes1.size(-1) == 4 or bboxes1.size(0) == 0)
    # assert (bboxes2.size(-1) == 4 or bboxes2.size(0) == 0)
    assert offset == 1 or offset == 0
    dim = bboxes1.size(-1) // 2

    rows = bboxes1.size(0)
    cols = bboxes2.size(0)
    if aligned:
        assert rows == cols

    if rows * cols == 0:
        return bboxes1.new(rows, 1) if aligned else bboxes1.new(rows, cols)

    if aligned:
        ious = bboxes1.new_zeros(rows)
    else:
        ious = bboxes1.new_zeros((rows, cols))
    if dim == 2:
        _C.bbox_overlaps_2d(
            bboxes1, bboxes2, ious, mode_flag, aligned, offset)
    else:
        _C.bbox_overlaps_3d(
            bboxes1, bboxes2, ious, mode_flag, aligned, offset)
    return ious