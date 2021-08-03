#pragma once
#include <torch/extension.h>

using namespace at;

#ifdef WITH_CUDA
void Affine2DForwardCUDAKernelLauncher(
    const at::Tensor features, const at::Tensor rois, const float spatial_scale,
    const int sample_ratio, const bool aligned, const int order,
    const int channels,
    const int height, const int width,
    const int num_rois,
    const int aligned_height, const int aligned_width,
    at::Tensor output);

void affine_2d_forward_cuda(
  Tensor features, Tensor rois, Tensor output,
  int aligned_height, int aligned_width,
  float spatial_scale, int sample_ratio,
  bool aligned, int order) {
  // Number of ROIs
  int num_rois = rois.size(0);
  int size_rois = rois.size(1);

  if (size_rois != 6) {
    AT_ERROR("wrong roi size! size_rois should be 6");
  }

  int num_channels = features.size(1);
  int data_height = features.size(2);
  int data_width = features.size(3);
  Affine2DForwardCUDAKernelLauncher(
      features, rois, spatial_scale, sample_ratio, aligned, order,
      num_channels, data_height, data_width, num_rois, aligned_height,
      aligned_width, output);
}
#endif

void affine_2d(Tensor input, Tensor rois, Tensor output,
               int aligned_height, int aligned_width,
               float spatial_scale, int sampling_ratio,
               bool aligned, int order) {
  if (input.device().is_cuda()) {
#ifdef WITH_CUDA
//    CHECK_CUDA_INPUT(input);
//    CHECK_CUDA_INPUT(rois);
//    CHECK_CUDA_INPUT(output);

    affine_2d_forward_cuda(input, rois, output, aligned_height,
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
