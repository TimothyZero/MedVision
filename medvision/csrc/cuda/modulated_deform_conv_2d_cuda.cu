// modified from mmdetection

#include "modulated_deform_conv_2d_cuda_kernel.cuh"

using namespace at;

#ifndef AT_CHECK
#define AT_CHECK TORCH_CHECK
#endif

void modulated_deformable_2d_im2col_cuda(
    const at::Tensor data_im, const at::Tensor data_offset, const at::Tensor data_mask,
    const int batch_size, const int channels,
    const int height_im, const int width_im,
    const int height_col, const int width_col,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int deformable_group, at::Tensor data_col)
{
  // num_axes should be smaller than block size
  const int channel_per_deformable_group = channels / deformable_group;
  const int num_kernels = channels * batch_size * height_col * width_col;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      data_im.scalar_type(), "modulated_deformable_im2col_gpu", ([&] {
        const scalar_t *data_im_ = data_im.data<scalar_t>();
        const scalar_t *data_offset_ = data_offset.data<scalar_t>();
        const scalar_t *data_mask_ = data_mask.data<scalar_t>();
        scalar_t *data_col_ = data_col.data<scalar_t>();

        modulated_deformable_im2col_gpu_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
            num_kernels, data_im_, data_offset_, data_mask_,
            height_im, width_im,
            kernel_h, kernel_w,
            pad_h, pad_w,
            stride_h, stride_w,
            dilation_h, dilation_w,
            channel_per_deformable_group, batch_size, channels, deformable_group,
            height_col, width_col, data_col_);
      }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in modulated_deformable_im2col_cuda: %s\n", cudaGetErrorString(err));
  }
}

void modulated_deformable_2d_col2im_cuda(
    const at::Tensor data_col, const at::Tensor data_offset, const at::Tensor data_mask,
    const int batch_size, const int channels,
    const int height_im, const int width_im,
    const int height_col, const int width_col,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int deformable_group, at::Tensor grad_im)
{

  const int channel_per_deformable_group = channels / deformable_group;
  const int num_kernels = channels * kernel_h * kernel_w * batch_size * height_col * width_col;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      data_col.scalar_type(), "modulated_deformable_col2im_gpu", ([&] {
        const scalar_t *data_col_ = data_col.data<scalar_t>();
        const scalar_t *data_offset_ = data_offset.data<scalar_t>();
        const scalar_t *data_mask_ = data_mask.data<scalar_t>();
        scalar_t *grad_im_ = grad_im.data<scalar_t>();

        modulated_deformable_col2im_gpu_kernel
        <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
            num_kernels, data_col_, data_offset_, data_mask_, channels,
            height_im, width_im,
            kernel_h, kernel_w,
            pad_h, pad_w,
            stride_h, stride_w,
            dilation_h, dilation_w,
            channel_per_deformable_group, batch_size, deformable_group,
            height_col, width_col, grad_im_);
      }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in modulated_deformable_col2im_cuda: %s\n", cudaGetErrorString(err));
  }
}

void modulated_deformable_2d_col2im_coord_cuda(
    const at::Tensor data_col, const at::Tensor data_im, const at::Tensor data_offset, const at::Tensor data_mask,
    const int batch_size, const int channels,
    const int height_im, const int width_im,
    const int height_col, const int width_col,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int deformable_group,
    at::Tensor grad_offset, at::Tensor grad_mask)
{
  const int channel_per_deformable_group = channels * kernel_h * kernel_w / deformable_group;
  const int num_kernels = batch_size * height_col * width_col * 2 * kernel_h * kernel_w * deformable_group;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      data_col.scalar_type(), "modulated_deformable_col2im_coord_gpu", ([&] {
        const scalar_t *data_col_ = data_col.data<scalar_t>();
        const scalar_t *data_im_ = data_im.data<scalar_t>();
        const scalar_t *data_offset_ = data_offset.data<scalar_t>();
        const scalar_t *data_mask_ = data_mask.data<scalar_t>();
        scalar_t *grad_offset_ = grad_offset.data<scalar_t>();
        scalar_t *grad_mask_ = grad_mask.data<scalar_t>();

        modulated_deformable_col2im_coord_gpu_kernel
        <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
            num_kernels, data_col_, data_im_, data_offset_, data_mask_, channels,
            height_im, width_im,
            kernel_h, kernel_w,
            pad_h, pad_w,
            stride_h, stride_w,
            dilation_h, dilation_w,
            channel_per_deformable_group,
            batch_size, 2 * kernel_h * kernel_w * deformable_group, deformable_group,
            height_col, width_col,
            grad_offset_, grad_mask_);
      }));
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in modulated_deformable_col2im_coord_cuda: %s\n", cudaGetErrorString(err));
  }
}


void modulated_deform_conv_2d_cuda_forward(
    at::Tensor input,
    at::Tensor weight,
    at::Tensor bias,
    at::Tensor ones,
    at::Tensor offset,
    at::Tensor mask,
    at::Tensor output,
    at::Tensor columns,
    int kernel_h,
    int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const int dilation_h,
    const int dilation_w,
    const int group,
    const int deformable_group,
    const bool with_bias) {
  AT_CHECK(input.is_contiguous(), "input tensor has to be contiguous");
  AT_CHECK(weight.is_contiguous(), "weight tensor has to be contiguous");
  at::DeviceGuard guard(input.device());

  const int batch = input.size(0);
  const int channels = input.size(1);
  const int height = input.size(2);
  const int width = input.size(3);

  const int channels_out = weight.size(0);
  const int channels_kernel = weight.size(1);
  const int kernel_h_ = weight.size(2);
  const int kernel_w_ = weight.size(3);

  if (kernel_h_ != kernel_h || kernel_w_ != kernel_w)
    AT_ERROR("Input shape and kernel shape wont match: (%d x %d vs %d x %d).", kernel_h_, kernel_w, kernel_h_, kernel_w_);

  if (channels != channels_kernel * group)
    AT_ERROR("Input shape and kernel channels wont match: (%d vs %d).",
             channels, channels_kernel * group);

  const int height_out = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int width_out = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

  if (ones.ndimension() != 2 ||
      ones.size(0) * ones.size(1) < height_out * width_out) {
    // Resize plane and fill with ones...
    ones = at::ones({height_out, width_out}, input.options());
  }

  // resize output
  output = output.view({batch, channels_out, height_out, width_out}).zero_();
  // resize temporary columns
  columns =
      at::zeros({channels * kernel_h * kernel_w, 1 * height_out * width_out},
                input.options());

  output = output.view({output.size(0), group, output.size(1) / group,
                        output.size(2), output.size(3)});

  for (int b = 0; b < batch; b++) {
    modulated_deformable_2d_im2col_cuda(
        input[b], offset[b], mask[b],
        1, channels,
        height, width,
        height_out, width_out,
        kernel_h, kernel_w,
        pad_h, pad_w,
        stride_h, stride_w,
        dilation_h, dilation_w,
        deformable_group, columns);

    // divide into group
    weight = weight.view({group, weight.size(0) / group, weight.size(1),
                          weight.size(2), weight.size(3)});
    columns = columns.view({group, columns.size(0) / group, columns.size(1)});

    for (int g = 0; g < group; g++) {
      output[b][g] = output[b][g]
                         .flatten(1)
                         .addmm_(weight[g].flatten(1), columns[g])
                         .view_as(output[b][g]);
    }

    weight = weight.view({weight.size(0) * weight.size(1), weight.size(2),
                          weight.size(3), weight.size(4)});
    columns =
        columns.view({columns.size(0) * columns.size(1), columns.size(2)});
  }

  output = output.view({output.size(0), output.size(1) * output.size(2),
                        output.size(3), output.size(4)});

  if (with_bias) {
    output += bias.view({1, bias.size(0), 1, 1});
  }
}

void modulated_deform_conv_2d_cuda_backward(
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
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    int group,
    int deformable_group,
    const bool with_bias) {
  AT_CHECK(input.is_contiguous(), "input tensor has to be contiguous");
  AT_CHECK(weight.is_contiguous(), "weight tensor has to be contiguous");
  at::DeviceGuard guard(input.device());

  const int batch = input.size(0);
  const int channels = input.size(1);
  const int height = input.size(2);
  const int width = input.size(3);

  const int channels_kernel = weight.size(1);
  const int kernel_h_ = weight.size(2);
  const int kernel_w_ = weight.size(3);

  if (kernel_h_ != kernel_h || kernel_w_ != kernel_w)
    AT_ERROR("Input shape and kernel shape wont match: (%d x %d vs %d x %d).",
             kernel_h_, kernel_w, kernel_h_, kernel_w_);
  if (channels != channels_kernel * group)
    AT_ERROR("Input shape and kernel channels wont match: (%d vs %d).",
             channels, channels_kernel * group);

  const int height_out = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int width_out = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

  if (ones.ndimension() != 2 ||
      ones.size(0) * ones.size(1) < height_out * width_out) {
    // Resize plane and fill with ones...
    ones = at::ones({height_out, width_out}, input.options());
  }

  grad_input = grad_input.view({batch, channels, height, width});
  columns = at::zeros({channels * kernel_h * kernel_w, height_out * width_out},
                      input.options());

  grad_output =
      grad_output.view({grad_output.size(0), group, grad_output.size(1) / group,
                        grad_output.size(2), grad_output.size(3)});

  for (int b = 0; b < batch; b++) {
    // divide int group
    columns = columns.view({group, columns.size(0) / group, columns.size(1)});
    weight = weight.view({group, weight.size(0) / group, weight.size(1),
                          weight.size(2), weight.size(3)});

    for (int g = 0; g < group; g++) {
      columns[g].addmm_(weight[g].flatten(1).transpose(0, 1),
                        grad_output[b][g].flatten(1), 0.0f, 1.0f);
    }

    columns =
        columns.view({columns.size(0) * columns.size(1), columns.size(2)});
    weight = weight.view({weight.size(0) * weight.size(1), weight.size(2),
                          weight.size(3), weight.size(4)});

    // gradient w.r.t. input coordinate data
    modulated_deformable_2d_col2im_coord_cuda(
        columns, input[b], offset[b], mask[b],
        1, channels,
        height, width,
        height_out, width_out,
        kernel_h, kernel_w,
        pad_h, pad_w,
        stride_h, stride_w,
        dilation_h, dilation_w,
        deformable_group, grad_offset[b],
        grad_mask[b]);
    // gradient w.r.t. input data
    modulated_deformable_2d_col2im_cuda(
        columns, offset[b], mask[b],
        1, channels,
        height, width,
        height_out, width_out,
        kernel_h, kernel_w,
        pad_h, pad_w,
        stride_h, stride_w,
        dilation_h, dilation_w,
        deformable_group, grad_input[b]);

    // gradient w.r.t. weight, dWeight should accumulate across the batch and
    // group
    modulated_deformable_2d_im2col_cuda(
        input[b], offset[b], mask[b],
        1, channels,
        height, width,
        height_out, width_out,
        kernel_h, kernel_w,
        pad_h, pad_w,
        stride_h, stride_w,
        dilation_h, dilation_w,
        deformable_group, columns);

    columns = columns.view({group, columns.size(0) / group, columns.size(1)});
    grad_weight = grad_weight.view({group, grad_weight.size(0) / group,
                                    grad_weight.size(1), grad_weight.size(2),
                                    grad_weight.size(3)});
    if (with_bias)
      grad_bias = grad_bias.view({group, grad_bias.size(0) / group});

    for (int g = 0; g < group; g++) {
      grad_weight[g] =
          grad_weight[g]
              .flatten(1)
              .addmm_(grad_output[b][g].flatten(1), columns[g].transpose(0, 1))
              .view_as(grad_weight[g]);
      if (with_bias) {
        grad_bias[g] =
            grad_bias[g]
                .view({-1, 1})
                .addmm_(grad_output[b][g].flatten(1), ones.view({-1, 1}))
                .view(-1);
      }
    }

    columns =
        columns.view({columns.size(0) * columns.size(1), columns.size(2)});
    grad_weight = grad_weight.view({grad_weight.size(0) * grad_weight.size(1),
                                    grad_weight.size(2), grad_weight.size(3),
                                    grad_weight.size(4)});
    if (with_bias)
      grad_bias = grad_bias.view({grad_bias.size(0) * grad_bias.size(1)});
  }
  grad_output = grad_output.view({grad_output.size(0) * grad_output.size(1),
                                  grad_output.size(2), grad_output.size(3),
                                  grad_output.size(4)});
}
