#ifndef RANDOM_NOISE_CUDA_KERNEL_CUH
#define RANDOM_NOISE_CUDA_KERNEL_CUH
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include "cuda_helpers.h"


template <typename scalar_t>
__global__ void random_noise_cuda_kernel(
    const int nthreads,
    scalar_t *bottom_data,
    float *noise_data,
    const int method,
    const float mean,
    const float std) {

  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    if (method == 0) {
      bottom_data[index] += mean + std * ((noise_data[index] - 0.5) / 0.5);
    } else if (method == 1) {
      bottom_data[index] += mean + std * noise_data[index];
    }
  }
}
#endif