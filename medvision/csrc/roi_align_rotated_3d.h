#pragma once
#include <torch/extension.h>

using namespace at;

#ifdef WITH_CUDA
void ROIAlignRotated3DForwardCUDAKernelLauncher(
    const at::Tensor features, const at::Tensor rois, const float spatial_scale,
    const int sample_ratio, const bool aligned, const int order,
    const int channels,
    const int depth, const int height, const int width,
    const int num_rois,
    const int aligned_depth, const int aligned_height, const int aligned_width,
    at::Tensor output);

void ROIAlignRotated3DBackwardCUDAKernelLauncher(
    const at::Tensor top_grad, const at::Tensor rois, const float spatial_scale,
    const int sample_ratio, const bool aligned, const int order,
    const int channels,
    const int depth, const int height, const int width,
    const int num_rois,
    const int aligned_depth, const int aligned_height, const int aligned_width,
    at::Tensor bottom_grad);

void roi_align_rotated_3d_forward_cuda(
  Tensor features, Tensor rois, Tensor output,
  int aligned_depth, int aligned_height, int aligned_width,
  float spatial_scale, int sample_ratio,
  bool aligned, int order) {
  // Number of ROIs
  int num_rois = rois.size(0);
  int size_rois = rois.size(1);

  if (size_rois != 10) {
    AT_ERROR("wrong roi size! size_rois should be 10");
  }

  int num_channels = features.size(1);
  int data_depth = features.size(2);
  int data_height = features.size(3);
  int data_width = features.size(4);
  ROIAlignRotated3DForwardCUDAKernelLauncher(
      features, rois, spatial_scale, sample_ratio, aligned, order,
      num_channels, data_depth, data_height, data_width, num_rois, aligned_depth, aligned_height,
      aligned_width, output);
}

void roi_align_rotated_3d_backward_cuda(
  Tensor top_grad, Tensor rois, Tensor bottom_grad,
  int aligned_depth, int aligned_height, int aligned_width,
  float spatial_scale, int sample_ratio, bool aligned, int order) {
  // Number of ROIs
  int num_rois = rois.size(0);
  int size_rois = rois.size(1);
  if (size_rois != 10) {
    AT_ERROR("wrong roi size! size_rois should be 10");
  }

  int num_channels = bottom_grad.size(1);
  int data_depth = bottom_grad.size(2);
  int data_height = bottom_grad.size(3);
  int data_width = bottom_grad.size(4);
  ROIAlignRotated3DBackwardCUDAKernelLauncher(
      top_grad, rois, spatial_scale, sample_ratio, aligned, order,
      num_channels, data_depth, data_height, data_width, num_rois, aligned_depth, aligned_height,
      aligned_width, bottom_grad);
}
#endif


void roi_align_rotated_3d_forward(Tensor input, Tensor rois, Tensor output,
                               int aligned_depth, int aligned_height, int aligned_width,
                               float spatial_scale, int sampling_ratio,
                               bool aligned, int order) {
  if (input.device().is_cuda()) {
#ifdef WITH_CUDA
//    CHECK_CUDA_INPUT(input);
//    CHECK_CUDA_INPUT(rois);
//    CHECK_CUDA_INPUT(output);

    roi_align_rotated_3d_forward_cuda(input, rois, output, aligned_depth, aligned_height,
                                   aligned_width, spatial_scale, sampling_ratio,
                                   aligned, order);
#else
    AT_ERROR("RoIAlignRotated is not compiled with GPU support");
#endif
  } else {
//    CHECK_CPU_INPUT(input);
//    CHECK_CPU_INPUT(rois);
//    CHECK_CPU_INPUT(output);
//    roi_align_rotated_forward_cpu(input, rois, output, aligned_height,
//                                  aligned_width, spatial_scale, sampling_ratio,
//                                  aligned, clockwise);
    AT_ERROR("RoIAlignRotated is not implemented on CPU");
  }
}

void roi_align_rotated_3d_backward(Tensor top_grad, Tensor rois,
                                Tensor bottom_grad, int aligned_depth, int aligned_height,
                                int aligned_width, float spatial_scale,
                                int sampling_ratio, bool aligned, int order) {
  if (top_grad.device().is_cuda()) {
#ifdef WITH_CUDA
//    CHECK_CUDA_INPUT(top_grad);
//    CHECK_CUDA_INPUT(rois);
//    CHECK_CUDA_INPUT(bottom_grad);

    roi_align_rotated_3d_backward_cuda(top_grad, rois, bottom_grad, aligned_depth, aligned_height,
                                    aligned_width, spatial_scale,
                                    sampling_ratio, aligned, order);
#else
    AT_ERROR("RoIAlignRotated is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("RoIAlignRotated is not implemented on CPU");
  }
}
