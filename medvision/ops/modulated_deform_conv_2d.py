import math

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

from medvision import _C


class ModulatedDeformConv2dFunction(Function):

    @staticmethod
    def forward(ctx,
                input,
                offset,
                mask,
                weight,
                bias=None,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                deformable_groups=1):
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.deformable_groups = deformable_groups
        ctx.with_bias = bias is not None
        if not ctx.with_bias:
            bias = input.new_empty(1)  # fake tensor
        if not input.is_cuda:
            raise NotImplementedError
        if weight.requires_grad or mask.requires_grad or offset.requires_grad \
                or input.requires_grad:
            ctx.save_for_backward(input, offset, mask, weight, bias)
        output = offset.new_empty(
            ModulatedDeformConv2dFunction._infer_shape(ctx, input, weight))
        ctx._bufs = [offset.new_empty(0), offset.new_empty(0)]
        if offset.dtype == torch.float16:
            input = input.half()
            bias = bias.half()
            weight = weight.half()
        _C.modulated_deform_conv_2d_forward(
            input, weight, bias, ctx._bufs[0], offset, mask, output,
            ctx._bufs[1], weight.shape[2], weight.shape[3], ctx.stride,
            ctx.stride, ctx.padding, ctx.padding, ctx.dilation, ctx.dilation,
            ctx.groups, ctx.deformable_groups, ctx.with_bias)
        ctx.save_for_backward(input, offset, mask, weight, bias)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        if not grad_output.is_cuda:
            raise NotImplementedError
        input, offset, mask, weight, bias = ctx.saved_tensors
        grad_input = torch.zeros_like(input)
        grad_offset = torch.zeros_like(offset)
        grad_mask = torch.zeros_like(mask)
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias)
        _C.modulated_deform_conv_2d_backward(
            input, weight, bias, ctx._bufs[0], offset, mask, ctx._bufs[1],
            grad_input, grad_weight, grad_bias, grad_offset, grad_mask,
            grad_output, weight.shape[2], weight.shape[3], ctx.stride,
            ctx.stride, ctx.padding, ctx.padding, ctx.dilation, ctx.dilation,
            ctx.groups, ctx.deformable_groups, ctx.with_bias)
        if not ctx.with_bias:
            grad_bias = None

        return (grad_input, grad_offset, grad_mask, grad_weight, grad_bias,
                None, None, None, None, None)

    @staticmethod
    def _infer_shape(ctx, input, weight):
        n = input.size(0)
        channels_out = weight.size(0)
        height, width = input.shape[2:4]
        kernel_h, kernel_w = weight.shape[2:4]
        height_out = (height + 2 * ctx.padding -
                      (ctx.dilation * (kernel_h - 1) + 1)) // ctx.stride + 1
        width_out = (width + 2 * ctx.padding -
                     (ctx.dilation * (kernel_w - 1) + 1)) // ctx.stride + 1
        return n, channels_out, height_out, width_out


modulated_deform_conv = ModulatedDeformConv2dFunction.apply


class ModulatedDeformConv2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 deformable_groups=1,
                 bias=True):
        super(ModulatedDeformConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.with_bias = bias

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups,
                         *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x, offset, mask):
        return modulated_deform_conv(x, offset, mask, self.weight, self.bias,
                                     self.stride, self.padding, self.dilation,
                                     self.groups, self.deformable_groups)

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


class ModulatedDeformConv2dPack(ModulatedDeformConv2d):

    def __init__(self, *args, **kwargs):
        super(ModulatedDeformConv2dPack, self).__init__(*args, **kwargs)

        self.conv_offset_mask = nn.Conv2d(
            self.in_channels,
            self.deformable_groups * 3 * self.kernel_size[0] *
            self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=_pair(self.stride),
            padding=_pair(self.padding),
            bias=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, x):
        out = self.conv_offset_mask(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return modulated_deform_conv(x, offset, mask, self.weight, self.bias,
                                     self.stride, self.padding, self.dilation,
                                     self.groups, self.deformable_groups)
