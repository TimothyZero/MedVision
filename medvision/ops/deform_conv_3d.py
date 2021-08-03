#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from torch.nn import init
import math
import torch
from torch import nn
from torch.autograd import Function
from torch.nn.modules.utils import _triple
from torch.autograd.function import once_differentiable

from medvision import _C


class DeformConv3dFunction(Function):
    @staticmethod
    def forward(ctx,
                input,
                offset,
                weight,
                bias,
                stride,
                padding,
                dilation,
                group,
                deformable_groups,
                im2col_step):
        ctx.stride = _triple(stride)
        ctx.padding = _triple(padding)
        ctx.dilation = _triple(dilation)
        ctx.kernel_size = _triple(weight.shape[2:5])
        ctx.group = group
        ctx.deformable_groups = deformable_groups
        ctx.im2col_step = im2col_step
        if offset.dtype == torch.float16:
            input = input.half()
            bias = bias.half()
            weight = weight.half()
        output = _C.deform_conv_3d_forward(input, weight, bias,
                                           offset,
                                           ctx.kernel_size[0], ctx.kernel_size[1], ctx.kernel_size[2],
                                           ctx.stride[0], ctx.stride[1], ctx.stride[2],
                                           ctx.padding[0], ctx.padding[1], ctx.padding[2],
                                           ctx.dilation[0], ctx.dilation[1], ctx.dilation[2],
                                           ctx.group,
                                           ctx.deformable_groups,
                                           ctx.im2col_step)
        ctx.save_for_backward(input, offset, weight, bias)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, offset, weight, bias = ctx.saved_tensors
        grad_input, grad_offset, grad_weight, grad_bias = \
            _C.deform_conv_3d_backward(input, weight,
                                       bias,
                                       offset,
                                       grad_output,
                                       ctx.kernel_size[0], ctx.kernel_size[1], ctx.kernel_size[2],
                                       ctx.stride[0], ctx.stride[1], ctx.stride[2],
                                       ctx.padding[0], ctx.padding[1], ctx.padding[2],
                                       ctx.dilation[0], ctx.dilation[1], ctx.dilation[2],
                                       ctx.group,
                                       ctx.deformable_groups,
                                       ctx.im2col_step)

        return (grad_input, grad_offset, grad_weight, grad_bias,
                None, None, None, None, None, None)


deform_conv = DeformConv3dFunction.apply


class DeformConv3d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation=1,
                 groups=1,
                 deformable_groups=1,
                 im2col_step=64,
                 bias=True):
        super(DeformConv3d, self).__init__()

        if in_channels % groups != 0:
            raise ValueError('in_channels {} must be divisible by groups {}'.format(in_channels, groups))
        if out_channels % groups != 0:
            raise ValueError('out_channels {} must be divisible by groups {}'.format(out_channels, groups))

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.padding = _triple(padding)
        self.dilation = _triple(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.im2col_step = im2col_step
        self.use_bias = bias

        self.weight = nn.Parameter(torch.Tensor(
            out_channels, in_channels // groups, *self.kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()
        if not self.use_bias:
            self.bias.requires_grad = False

    def reset_parameters(self):
        n = self.in_channels
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input, offset):
        assert 3 * self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2] == \
               offset.shape[1]
        return deform_conv(input, offset,
                           self.weight,
                           self.bias,
                           self.stride,
                           self.padding,
                           self.dilation,
                           self.groups,
                           self.deformable_groups,
                           self.im2col_step)

    def __repr__(self):
        s = self.__class__.__name__
        s += f'(in_channels={self.in_channels}, '
        s += f'out_channels={self.out_channels}, '
        s += f'kernel_size={self.kernel_size}, '
        s += f'stride={self.stride}, '
        s += f'padding={self.padding}, '
        s += f'dilation={self.dilation}, '
        s += f'groups={self.groups}, '
        s += f'deform_groups={self.deformable_groups})'
        return s


class DeformConv3dPack(DeformConv3d):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 deformable_groups=1,
                 im2col_step=64,
                 bias=True,
                 lr_mult=0.1):
        super(DeformConv3dPack, self).__init__(in_channels, out_channels,
                                               kernel_size, stride, padding, dilation, groups, deformable_groups,
                                               im2col_step, bias)

        out_channels = self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]
        self.conv_offset = nn.Conv3d(self.in_channels,
                                     out_channels,
                                     kernel_size=self.kernel_size,
                                     stride=self.stride,
                                     padding=self.padding,
                                     bias=True)
        self.conv_offset.lr_mult = lr_mult
        self.init_offset()

    def init_offset(self):
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

    def forward(self, input):
        offset = self.conv_offset(input)
        return deform_conv(input, offset,
                           self.weight,
                           self.bias,
                           self.stride,
                           self.padding,
                           self.dilation,
                           self.groups,
                           self.deformable_groups,
                           self.im2col_step)
