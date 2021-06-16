"""
ROIAlign implementation from pytorch framework
(https://github.com/pytorch/vision/blob/master/torchvision/ops/roi_align.py on Nov 14 2019)

adapted for 3D support without additional python function interface (only cpp function interface).
copied from MIC-DKFZ/RegRCNN
"""

import torch
from torch import nn
from torchvision.ops._utils import convert_boxes_to_roi_format

from medvision import _C


def roi_align_2d(
        input: torch.Tensor,
        boxes: torch.Tensor,
        output_size,
        spatial_scale: float = 1.0,
        sampling_ratio: int = -1
) -> torch.Tensor:
    """
    Performs Region of Interest (RoI) Align operator described in Mask R-CNN

    Arguments:
        input: (Tensor[N, C, H, W]), input tensor
        boxes: (Tensor[K, 5] or List[Tensor[L, 4]]), the box coordinates in (y1, x1, y2, x2)
            NOTE: the order of box coordinates, (y1, x1, y2, x2), is swapped w.r.t. to the order in the
                original torchvision implementation (which requires (x1, y1, x2, y2)).
            format where the regions will be taken from. If a single Tensor is passed,
            then the first column should contain the batch index. If a list of Tensors
            is passed, then each Tensor will correspond to the boxes for an element i
            in a batch
        output_size: (Tuple[int, int]) the size of the output after the cropping
            is performed, as (height, width)
        spatial_scale: (float) a scaling factor that maps the input coordinates to
            the box coordinates. Default: 1.0
        sampling_ratio: (int) number of sampling points in the interpolation grid
            used to compute the output value of each pooled output bin. If > 0,
            then exactly sampling_ratio x sampling_ratio grid points are used. If
            <= 0, then an adaptive number of grid points are used (computed as
            ceil(roi_width / pooled_w), and likewise for height). Default: -1

    Returns:
        output (Tensor[K, C, output_size[0], output_size[1]])
    """
    rois = boxes
    if isinstance(rois, list):
        rois = convert_boxes_to_roi_format(rois)
    return _C.roi_align_2d(input, rois, spatial_scale, output_size[0], output_size[1], sampling_ratio)


def roi_align_3d(
        input: torch.Tensor,
        boxes: torch.Tensor,
        output_size,
        spatial_scale: float = 1.0,
        sampling_ratio: int = -1
) -> torch.Tensor:
    """
    Performs Region of Interest (RoI) Align operator described in Mask R-CNN for 3-dim input.

    Arguments:
        input (Tensor[N, C, H, W, D]): input tensor
        boxes (Tensor[K, 7] or List[Tensor[L, 6]]): the box coordinates in (y1, x1, y2, x2, z1, z2).
            NOTE: the order of x, y box coordinates, (y1, x1, y2, x2), is swapped w.r.t. to the order in the
                original torchvision implementation (which requires (x1, y1, x2, y2)).
            format where the regions will be taken from. If a single Tensor is passed,
            then the first column should contain the batch index. If a list of Tensors
            is passed, then each Tensor will correspond to the boxes for an element i
            in a batch
        output_size (int or Tuple[int, int, int]): the size of the output after the cropping
            is performed, as (height, width, depth)
        spatial_scale (float): a scaling factor that maps the input coordinates to
            the box coordinates. Default: 1.0
        sampling_ratio (int): number of sampling points in the interpolation grid
            used to compute the output value of each pooled output bin. If > 0,
            then exactly sampling_ratio x sampling_ratio grid points are used. If
            <= 0, then an adaptive number of grid points are used (computed as
            ceil(roi_width / pooled_w), and likewise for height). Default: -1

    Returns:
        output (Tensor[K, C, output_size[0], output_size[1], output_size[2]])
    """
    rois = boxes
    if isinstance(rois, list):
        rois = convert_boxes_to_roi_format(rois)
    return _C.roi_align_3d(input, rois, spatial_scale, output_size[0], output_size[1], output_size[2],
                           sampling_ratio)


class RoIAlign(nn.Module):
    """
    Performs Region of Interest (RoI) Align operator described in Mask R-CNN for 2- or 3-dim input.

    Arguments:
        output_size (int or Tuple[int, int(, int)]): the size of the output after the cropping
            is performed, as (height, width(, depth))
        spatial_scale (float): a scaling factor that maps the input coordinates to
            the box coordinates. Default: 1.0
        sampling_ratio (int): number of sampling points in the interpolation grid
            used to compute the output value of each pooled output bin. If > 0,
            then exactly sampling_ratio x sampling_ratio grid points are used. If
            <= 0, then an adaptive number of grid points are used (computed as
            ceil(roi_width / pooled_w), and likewise for height (and depth)). Default: -1

    Returns:
        output (Tensor[K, C, output_size[0], output_size[1](, output_size[2])])
    """

    def __init__(self, output_size, spatial_scale=1., sampling_ratio=-1):
        super(RoIAlign, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
        self.dim = len(self.output_size)

        if self.dim == 2:
            self.roi_align = roi_align_2d
        elif self.dim == 3:
            self.roi_align = roi_align_3d
        else:
            raise Exception("Tried to init RoIAlign module with incorrect output size: {}".format(self.output_size))

    def forward(self, feature: torch.Tensor, rois: torch.Tensor) -> torch.Tensor:
        if self.dim == 2:
            # ATTENTION : Input shape is HW => yx, required y1,x1,y2,x2
            # but my coord is x1,y1,x2,y2
            if isinstance(rois, list):
                rois = [roi[:, [1, 0, 3, 2]] for roi in rois]
            else:
                rois = rois[:, [0, 2, 1, 4, 3]]
        else:
            # ATTENTION : Input shape is HWD => yxz, required y1,x1,y2,x2,z1,z2,  <----------------
            # but my code is :  shape is DHW => zyx, coord is x1,y1,z1,x2,y2,z2                    |
            # equals to z1,x1,y1,z2,x2,y2 based on HWD => yxz      ----------> swap axes ------->  |
            if isinstance(rois, list):
                rois = [roi[:, [2, 1, 5, 4, 0, 3]] for roi in rois]
            else:
                rois = rois[:, [0, 3, 2, 6, 5, 1, 4]]
        return self.roi_align(feature, rois, self.output_size, self.spatial_scale, self.sampling_ratio)

    def __repr__(self):
        repr_str = self.__class__.__name__ + '('
        repr_str += 'output_size=' + str(self.output_size)
        repr_str += ', spatial_scale=' + str(self.spatial_scale)
        repr_str += ', sampling_ratio=' + str(self.sampling_ratio)
        repr_str += ', dimension=' + str(self.dim)
        repr_str += ')'
        return repr_str
