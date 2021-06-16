
import torch
from medvision import _C


def nms_nd(dets, iou_threshold):
    """
    Performs non-maximum suppression (NMS) on the boxes according
    to their intersection-over-union (IoU).

    NMS iteratively removes lower scoring boxes which have an
    IoU greater than iou_threshold with another (higher scoring)
    box.

    Parameters
    ----------
    boxes : Tensor[N, 4] for 2D or Tensor[N,6] for 3D.
        boxes to perform NMS on. They
        are expected to be in (y1, x1, y2, x2(, z1, z2)) format
    scores : Tensor[N]
        scores for each one of the boxes
    iou_threshold : float
        discards all overlapping
        boxes with IoU < iou_threshold

    Returns
    -------
    keep : Tensor
        int64 tensor with the indices
        of the elements that have been kept
        by NMS, sorted in decreasing order of scores
    """
    dim = dets.shape[-1] // 2
    boxes = dets[:, :-1]
    scores = dets[:, -1]
    if dim == 2:
        return _C.nms_2d(boxes, scores, iou_threshold), dets
    else:
        return _C.nms_3d(boxes, scores, iou_threshold), dets
