#pragma once

#include <cuda.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <THC/THCAtomics.cuh>

#define CUDA_1D_KERNEL_LOOP(i, n)                                \
  for (int i = (blockIdx.x * blockDim.x) + threadIdx.x; i < (n); \
       i += (blockDim.x * gridDim.x))

using at::Half;
using at::Tensor;
using phalf = at::Half;

const int CUDA_NUM_THREADS = 256;
const int THREADS_PER_BLOCK = CUDA_NUM_THREADS;

inline int GET_BLOCKS(const int N)
{
    int optimal_block_num = (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
    int max_block_num = 4096;
    return min(optimal_block_num, max_block_num);
}