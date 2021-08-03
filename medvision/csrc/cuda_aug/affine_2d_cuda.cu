#include "affine_2d_cuda_kernel.cuh"

using namespace at;

void Affine2DForwardCUDAKernelLauncher(
    const at::Tensor features, const at::Tensor rois, const float spatial_scale,
    const int sample_num, const bool aligned, const int order,
    const int channels,
    const int height, const int width,
    const int num_rois,
    const int pooled_height, const int pooled_width,
    at::Tensor output) {
  const int output_size = num_rois * pooled_height * pooled_width * channels;
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      features.type(), "Affine2DLaucherForward", ([&] {
        const scalar_t *bottom_data = features.contiguous().data<scalar_t>();
        const scalar_t *rois_data = rois.contiguous().data<scalar_t>();
        scalar_t *top_data = output.contiguous().data<scalar_t>();

        affine_2d_forward_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
                output_size, bottom_data, rois_data, scalar_t(spatial_scale),
                sample_num, aligned, order, channels,
                height, width,
                pooled_height, pooled_width,
                top_data);
      }));

  AT_CUDA_CHECK(cudaGetLastError());
}