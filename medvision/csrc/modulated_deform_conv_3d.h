#pragma once
#include <cstdio>
#include <cstring>
#include <torch/extension.h>

void modulated_deform_conv_3d_cuda_forward(
    at::Tensor input,
    at::Tensor weight,
    at::Tensor bias,
    at::Tensor ones,
    at::Tensor offset,
    at::Tensor mask,
    at::Tensor output,
    at::Tensor columns,
    int kernel_d,
    int kernel_h,
    int kernel_w,
    const int stride_d,
    const int stride_h,
    const int stride_w,
    const int pad_d,
    const int pad_h,
    const int pad_w,
    const int dilation_d,
    const int dilation_h,
    const int dilation_w,
    const int group,
    const int deformable_group,
    const bool with_bias);

void modulated_deform_conv_3d_cuda_backward(
    at::Tensor input,
    at::Tensor weight,
    at::Tensor bias,
    at::Tensor ones,
    at::Tensor offset,
    at::Tensor mask,
    at::Tensor columns,
    at::Tensor grad_input,
    at::Tensor grad_weight,
    at::Tensor grad_bias,
    at::Tensor grad_offset,
    at::Tensor grad_mask,
    at::Tensor grad_output,
    int kernel_d,
    int kernel_h,
    int kernel_w,
    int stride_d,
    int stride_h,
    int stride_w,
    int pad_d,
    int pad_h,
    int pad_w,
    int dilation_d,
    int dilation_h,
    int dilation_w,
    int group,
    int deformable_group,
    const bool with_bias);
    
void modulated_deform_conv_3d_forward(
    at::Tensor input, at::Tensor weight, at::Tensor bias, at::Tensor ones,
    at::Tensor offset, at::Tensor mask, at::Tensor output, at::Tensor columns,
    int kernel_d, int kernel_h, int kernel_w,
    const int stride_d, const int stride_h, const int stride_w,
    const int pad_d, const int pad_h, const int pad_w,
    const int dilation_d, const int dilation_h, const int dilation_w,
    const int group, const int deformable_group,
    const bool with_bias)
{
    if (input.type().is_cuda())
    {
#ifdef WITH_CUDA
    return modulated_deform_conv_3d_cuda_forward(input, weight, bias, ones,
                                          offset, mask, output, columns,
                                          kernel_d, kernel_h, kernel_w,
                                          stride_d, stride_h, stride_w,
                                          pad_d, pad_h, pad_w,
                                          dilation_d, dilation_h, dilation_w,
                                          group, deformable_group,
                                          with_bias);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    AT_ERROR("Not implemented on the CPU");
}

void modulated_deform_conv_3d_backward(
    at::Tensor input, at::Tensor weight, at::Tensor bias, at::Tensor ones,
    at::Tensor offset, at::Tensor mask, at::Tensor columns,
    at::Tensor grad_input, at::Tensor grad_weight, at::Tensor grad_bias,
    at::Tensor grad_offset, at::Tensor grad_mask, at::Tensor grad_output,
    int kernel_d, int kernel_h, int kernel_w,
    const int stride_d, const int stride_h, const int stride_w,
    const int pad_d, const int pad_h, const int pad_w,
    const int dilation_d, const int dilation_h, const int dilation_w,
    int group, int deformable_group,
    const bool with_bias)
{
    if (input.type().is_cuda())
    {
#ifdef WITH_CUDA
    return modulated_deform_conv_3d_cuda_backward(input, weight, bias, ones,
                                           offset, mask, columns,
                                           grad_input, grad_weight, grad_bias,
                                           grad_offset, grad_mask, grad_output,
                                           kernel_d, kernel_h, kernel_w,
                                           stride_d, stride_h, stride_w,
                                           pad_d, pad_h, pad_w,
                                           dilation_d, dilation_h, dilation_w,
                                           group, deformable_group,
                                           with_bias);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    AT_ERROR("Not implemented on the CPU");
}