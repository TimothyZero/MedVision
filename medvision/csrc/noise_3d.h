#pragma once
#include <torch/extension.h>

using namespace at;

#ifdef WITH_CUDA
void RandomNoise3DForwardCUDAKernelLauncher(
    at::Tensor image,
    const int method,
    const float mean,
    const float std,
    const int batch,
    const int channels,
    const int depth,
    const int height,
    const int width);

void random_noise_3d_cuda(
  Tensor image, int method, float mean, float std) {

  int num_batches = image.size(0);
  int num_channels = image.size(1);
  int data_depth = image.size(2);
  int data_height = image.size(3);
  int data_width = image.size(4);
  RandomNoise3DForwardCUDAKernelLauncher(
      image,
      method, mean, std,
      num_batches, num_channels,
      data_depth, data_height, data_width);
}

#endif

void noise_3d(
  Tensor image, int method, float mean, float std) {
  if (image.device().is_cuda()) {

#ifdef WITH_CUDA
    random_noise_3d_cuda(image, method, mean, std);
#endif
  } else {
    AT_ERROR("random_noise is not implemented on CPU");
  }
}