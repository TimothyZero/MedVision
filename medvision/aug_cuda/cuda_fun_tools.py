# -*- coding:utf-8 -*-
import torch
from medvision import _C


def affine_2d(features,
              rois,
              out_size,
              spatial_scale,
              sampling_ratio=0,
              aligned=True,
              order=1):
    if isinstance(out_size, int):
        out_h = out_size
        out_w = out_size
    elif isinstance(out_size, tuple):
        assert len(out_size) == 2
        assert isinstance(out_size[0], int)
        assert isinstance(out_size[1], int)
        out_h, out_w = out_size
    else:
        raise TypeError(
            '"out_size" must be an integer or tuple of integers')
    assert features.dtype in [torch.float32, torch.float16], \
        f'input must be float16 or float32 nut get {features.dtype}'
    assert order in [0, 1, 3], f'order {order} is not supported!'

    batch_size, num_channels, data_height, data_width = features.size()
    num_rois = rois.size(0)

    output = features.new_zeros(num_rois, num_channels, out_h, out_w)
    _C.affine_2d(
        features,
        rois.type(features.type()),
        output,
        out_h,
        out_w,
        spatial_scale,
        sampling_ratio,
        aligned,
        order)
    return output


def affine_3d(features,
              rois,
              out_size,
              spatial_scale,
              sampling_ratio=0,
              aligned=True,
              order=1):
    # clockwise is not used in 3d
    if isinstance(out_size, int):
        out_d = out_size
        out_h = out_size
        out_w = out_size
    elif isinstance(out_size, tuple):
        assert len(out_size) == 3
        assert isinstance(out_size[0], int)
        assert isinstance(out_size[1], int)
        assert isinstance(out_size[2], int)
        out_d, out_h, out_w = out_size
    else:
        raise TypeError(
            '"out_size" must be an integer or tuple of integers')
    assert features.dtype in [torch.float32, torch.float16], \
        f'input must be float16 or float32 nut get {features.dtype}'
    assert order in [0, 1, 3], f'order {order} is not supported!'

    batch_size, num_channels, data_depth, data_height, data_width = features.size()
    num_rois = rois.size(0)

    output = features.new_zeros(num_rois, num_channels, out_d, out_h, out_w)
    _C.affine_3d(
        features,
        rois.type(features.type()),
        output,
        out_d,
        out_h,
        out_w,
        spatial_scale,
        sampling_ratio,
        aligned,
        order)
    return output


def apply_offset_2d(img, offset, order=1):
    """
    image : b, c, d, h, w
    offset : b, 2, d, h, w
    """
    assert img.shape[2:] == offset.shape[2:]
    assert offset.shape[1] == 2

    channels = img.shape[1]
    kernel_size = [1, 1]
    stride = [1, 1]
    padding = [0, 0]
    dilation = [1, 1]
    group = 1
    deformable_groups = 1
    im2col_step = 64

    weight = torch.eye(channels, channels).unsqueeze(-1).unsqueeze(-1).cuda()
    bias = torch.zeros(channels).cuda()
    offset = offset.cuda()
    if img.dtype == torch.float16:
        offset = offset.half()
        bias = bias.half()
        weight = weight.half()
    output = _C.deform_2d(img.contiguous(),
                          weight.contiguous(),
                          bias.contiguous(),
                          offset.contiguous(),
                          kernel_size[0], kernel_size[1],
                          stride[0], stride[1],
                          padding[0], padding[1],
                          dilation[0], dilation[1],
                          group,
                          deformable_groups,
                          im2col_step,
                          order)
    assert img.shape == output.shape
    return output


def apply_offset_3d(img, offset, order=1):
    assert img.shape[2:] == offset.shape[2:]
    assert offset.shape[1] == 3

    channels = img.shape[1]
    kernel_size = [1, 1, 1]
    stride = [1, 1, 1]
    padding = [0, 0, 0]
    dilation = [1, 1, 1]
    group = 1
    deformable_groups = 1
    im2col_step = 64

    weight = torch.eye(channels, channels).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).cuda()
    bias = torch.zeros(channels).cuda()
    offset = offset.cuda()
    if img.dtype == torch.float16:
        offset = offset.half()
        bias = bias.half()
        weight = weight.half()
    output = _C.deform_3d(img.contiguous(),
                          weight.contiguous(),
                          bias.contiguous(),
                          offset.contiguous(),
                          kernel_size[0], kernel_size[1], kernel_size[2],
                          stride[0], stride[1], stride[2],
                          padding[0], padding[1], padding[2],
                          dilation[0], dilation[1], dilation[2],
                          group,
                          deformable_groups,
                          im2col_step,
                          order)
    assert img.shape == output.shape, f"input is {img.shape}, out is {output.shape}"
    return output


def random_noise_2d(img, method, mean=0., std=1., inplace=False):
    """
    method:
        0 : uniform , U[-0.5, 0.5]
        1 : normal, N(0, 1)
    mean:
        std * gen_noise + mean
    """
    if not inplace:
        out = img.clone()
        _C.noise_2d(out, method, mean, std)
        return out
    else:
        _C.noise_2d(img, method, mean, std)
        return img


def random_noise_3d(img, method, mean=0., std=1., inplace=False):
    if not inplace:
        out = img.clone()
        _C.noise_3d(out, method, mean, std)
        return out
    else:
        _C.noise_3d(img, method, mean, std)
        return img
