#include <curand.h>
#include <curand_kernel.h>

#include "random_noise_cuda_kernel.cuh"

using namespace at;

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    }} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    }} while(0)

void RandomNoise3DForwardCUDAKernelLauncher(
  at::Tensor image,
  const int method, const float mean, const float std,
  const int batch, const int channels,
  const int depth, const int height, const int width) {
  const int output_size = batch * depth * height * width * channels;

  curandGenerator_t gen;
  float *noise_data;

  /* Allocate n floats on device */
  CUDA_CALL(cudaMalloc((void **)&noise_data, output_size * sizeof(float)));
  /* Create pseudo-random number generator */
  CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  /* Set seed */
  CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));
  /* Generate n floats on device */
  if (method == 0){
    CURAND_CALL(curandGenerateUniform(gen, noise_data, output_size));
  } else if (method == 1){
    CURAND_CALL(curandGenerateNormal(gen, noise_data, output_size, 0., 1.0));
  }

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      image.type(), "RandomNoise3DForwardCUDAKernel", ([&] {
        scalar_t *bottom_data = image.data<scalar_t>();

        random_noise_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
                output_size,
                bottom_data, noise_data,
                method, mean, std);
      }));

  /* Cleanup */
  CURAND_CALL(curandDestroyGenerator(gen));
  CUDA_CALL(cudaFree(noise_data));

  AT_CUDA_CHECK(cudaGetLastError());
}