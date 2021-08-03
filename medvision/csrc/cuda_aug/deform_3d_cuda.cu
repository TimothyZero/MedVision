// copied from https://github.com/XinyiYing/D3Dnet

#include "deform_3d_cuda_kernel.cuh"

using namespace at;

template <typename scalar_t>
void deformable_3d_im2col_cuda(cudaStream_t stream,
  const scalar_t* data_im, const scalar_t* data_offset,
  const int batch_size, const int channels,
  const int depth_im, const int height_im, const int width_im,
  const int depth_col, const int height_col, const int width_col,
  const int kernel_d, const int kernel_h, const int kernel_w,
  const int pad_d, const int pad_h, const int pad_w,
  const int stride_d, const int stride_h, const int stride_w,
  const int dilation_d, const int dilation_h, const int dilation_w,
  const int deformable_group, scalar_t* data_col,
  const bool ismask)
{
  // num_axes should be smaller than block size
  const int channel_per_deformable_group = channels / deformable_group;
  const int num_kernels = channels * batch_size * depth_col * height_col * width_col;

  deformable_im2col_gpu_kernel<scalar_t>
  <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(
    num_kernels, data_im, data_offset,
    depth_im, height_im, width_im,
    kernel_d, kernel_h, kernel_w,
    pad_d, pad_h, pad_w,
    stride_d, stride_h, stride_w,
    dilation_d, dilation_h, dilation_w,
    channel_per_deformable_group,
    batch_size, channels, deformable_group,
    depth_col, height_col, width_col, data_col, ismask);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in deformable_im2col_cuda: %s\n", cudaGetErrorString(err));
  }
}

at::Tensor
deform_3d_cuda_forward(const at::Tensor &input,
                    const at::Tensor &weight,
                    const at::Tensor &bias,
                    const at::Tensor &offset,
                    const int kernel_d,
                    const int kernel_h,
                    const int kernel_w,
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
                    const int im2col_step,
                    const bool ismask)
{
    // THCAssertSameGPU(THCudaTensor_checkGPU(state, 5, input, weight, bias, offset, mask));

    AT_ASSERTM(input.is_contiguous(), "input tensor has to be contiguous");
    AT_ASSERTM(weight.is_contiguous(), "weight tensor has to be contiguous");

    AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(weight.type().is_cuda(), "weight must be a CUDA tensor");
    AT_ASSERTM(bias.type().is_cuda(), "bias must be a CUDA tensor");
    AT_ASSERTM(offset.type().is_cuda(), "offset must be a CUDA tensor");

    const int batch = input.size(0);
    const int channels = input.size(1);
    const int depth = input.size(2);
    const int height = input.size(3);
    const int width = input.size(4);

    const int channels_out = weight.size(0);
    const int channels_kernel = weight.size(1);
    const int kernel_d_ = weight.size(2);
    const int kernel_h_ = weight.size(3);
    const int kernel_w_ = weight.size(4);

    const int im2col_step_ = std::min(batch, im2col_step);

    AT_ASSERTM(batch % im2col_step_ == 0, "batch(%d) must divide im2col_step(%d)", batch, im2col_step_);

    AT_ASSERTM((channels % group == 0) && (channels_out % group == 0),
        "channels(%d) and channels_out(%d) must divide group(%d)", channels, channels_out, group);

    // printf("Kernels: %d %d %d %d %d %d\n", kernel_d_, kernel_h_, kernel_w_, kernel_d, kernel_w, kernel_h);
    // printf("Channels: %d %d\n", channels, channels_kernel);
    // printf("Channels: %d %d\n", channels_out, channels_kernel);

    AT_ASSERTM(kernel_h_ == kernel_h && kernel_w_ == kernel_w && kernel_d_ == kernel_d,
               "Input shape and kernel shape wont match: (%d x %d x %d vs %d x %d x %d).", kernel_h, kernel_w, kernel_d, kernel_h_, kernel_w_, kernel_d_);

    AT_ASSERTM(channels == (channels_kernel * group),
               "Input shape and kernel channels wont match: (%d vs %d).", channels, channels_kernel * group);

    const int depth_out = (depth + 2 * pad_d - (dilation_d * (kernel_d - 1) + 1)) / stride_d + 1;
    const int height_out = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int width_out = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

    auto output = at::empty({batch * depth_out * height_out * width_out, channels_out}, input.options());

    // prepare group weight and bias
    auto weight_g = weight.view({group, channels_out/group, channels_kernel, kernel_d, kernel_h, kernel_w});
    auto bias_g = bias.view({group, channels_out/group});

    // define alias for easy use
    const int batch_n = im2col_step_;
    const int per_input_size = channels * depth * height * width;
    const int per_offset_size = offset.size(1) * offset.size(2) * offset.size(3) * offset.size(4);
    auto output_n = output.view({batch/im2col_step_, batch_n * depth_out * height_out * width_out, channels_out});
    for (int n = 0; n < batch/im2col_step_; ++n)
    {
        auto columns = at::empty({channels * kernel_d * kernel_h * kernel_w, batch_n * depth_out * height_out * width_out}, input.options());
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "deform_conv_forward_cuda", ([&] {
            deformable_3d_im2col_cuda(at::cuda::getCurrentCUDAStream(),
                                             input.data<scalar_t>() + n * im2col_step_ * per_input_size,
                                             offset.data<scalar_t>() + n * im2col_step_ * per_offset_size,
                                             batch_n, channels, depth, height, width,
                                             depth_out, height_out, width_out, kernel_d, kernel_h, kernel_w,
                                             pad_d, pad_h, pad_w, stride_d, stride_h, stride_w, dilation_d, dilation_h, dilation_w,
                                             deformable_group,
                                             columns.data<scalar_t>(), ismask);

        }));

        // auto columns_m = columns.t();
        // auto weight_m = weight.view({channels_out, channels_kernel * kernel_d * kernel_h * kernel_w}).t();
        // output = at::addmm(bias, columns_m, weight_m);
        auto columns_g = columns.view({group, channels/group * kernel_d * kernel_h * kernel_w, batch_n * depth_out * height_out * width_out});
        auto output_g = output_n.select(0, n).view({batch_n * depth_out * height_out * width_out, group, channels_out/group});
        for (int g = 0; g < group; ++g)
        {
            auto columns_gm = columns_g.select(0, g).t();
            auto weight_gm = weight_g.select(0, g).view({channels_out/group, channels_kernel * kernel_d * kernel_h * kernel_w}).t();
            auto output_m = at::addmm(bias_g.select(0, g), columns_gm, weight_gm);
            output_g.select(1, g) = output_m.view({batch_n * depth_out * height_out * width_out, channels_out/group});
        }

    }

    output = output.view({batch, depth_out, height_out, width_out, channels_out}).permute({0, 4, 1, 2, 3}).contiguous();

    return output;
}
