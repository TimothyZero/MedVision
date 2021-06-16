#include <vector>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THCAtomics.cuh>
#include <cstdio>
#include <algorithm>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>


// modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/blob/pytorch_1.0.0/src/cuda/deform_conv_cuda.cu

// copied from https://github.com/XinyiYing/D3Dnet

#define CUDA_KERNEL_LOOP(i, n)                          \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;   \
      i < (n);                                          \
      i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 256;
inline int GET_BLOCKS(const int N)
{
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

// deformable DCNv2 3d

template <typename scalar_t>
__device__ scalar_t deformable_im2col_bilinear(const scalar_t *bottom_data, const int data_heigth, const int data_width,
                                      const int depth, const int height, const int width, scalar_t d, scalar_t h, scalar_t w)
{
  int d_low = floor(d);
  int h_low = floor(h);
  int w_low = floor(w);
  int d_high = d_low + 1;
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  scalar_t ld = d - d_low;
  scalar_t lh = h - h_low;
  scalar_t lw = w - w_low;
  scalar_t hd = 1 - ld, hh = 1 - lh, hw = 1 - lw;

  scalar_t v1 = 0;
  if (d_low >= 0 && h_low >= 0 && w_low >= 0 )
    v1 = bottom_data[d_low * data_heigth * data_width + h_low * data_width + w_low];
  scalar_t v2 = 0;
  if (d_low >= 0 && h_low >= 0 && w_high <= width - 1)
    v2 = bottom_data[d_low * data_heigth * data_width + h_low * data_width + w_high];
  scalar_t v3 = 0;
  if (d_low >= 0 && h_high <= height - 1 && w_low >= 0)
    v3 = bottom_data[d_low * data_heigth * data_width + h_high * data_width + w_low];
  scalar_t v4 = 0;
 if (d_low >= 0 && h_high <= height - 1 && w_high <= width - 1)
    v4 = bottom_data[d_low * data_heigth * data_width + h_high * data_width + w_high];
  scalar_t v5 = 0;
  if (d_high <= depth - 1 && h_low >= 0 && w_low >= 0 )
    v5 = bottom_data[d_high * data_heigth * data_width + h_low * data_width + w_low];
  scalar_t v6 = 0;
  if (d_high <= depth - 1 && h_low >= 0 && w_high <= width - 1)
    v6 = bottom_data[d_high * data_heigth * data_width + h_low * data_width + w_high];
  scalar_t v7 = 0;
  if (d_high <= depth - 1 && h_high <= height - 1 && w_low >= 0)
    v7 = bottom_data[d_high * data_heigth * data_width + h_high * data_width + w_low];
  scalar_t v8 = 0;
 if (d_high <= depth - 1 && h_high <= height - 1 && w_high <= width - 1)
    v8 = bottom_data[d_high * data_heigth * data_width + h_high * data_width + w_high];

  scalar_t w1 = hd * hh * hw, w2 = hd * hh * lw, w3 = hd * lh * hw, w4 = hd * lh * lw;
  scalar_t w5 = ld * hh * hw, w6 = ld * hh * lw, w7 = ld * lh * hw, w8 = ld * lh * lw;

  scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4 + w5 * v5 + w6 * v6 + w7 * v7 + w8 * v8);
  return val;
}

template <typename scalar_t>
__device__ scalar_t deformable_get_gradient_weight(scalar_t argmax_d, scalar_t argmax_h, scalar_t argmax_w,
                                          const int d, const int h, const int w, const int depth, const int height, const int width)
{
  if (argmax_d <= -1 || argmax_d >= depth || argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 || argmax_w >= width)
  {
    //empty
    return 0;
  }

  int argmax_d_low = floor(argmax_d);
  int argmax_h_low = floor(argmax_h);
  int argmax_w_low = floor(argmax_w);
  int argmax_d_high = argmax_d_low + 1;
  int argmax_h_high = argmax_h_low + 1;
  int argmax_w_high = argmax_w_low + 1;

  scalar_t weight = 0;
  if (d == argmax_d_low && h == argmax_h_low && w == argmax_w_low)
    weight = (d + 1 - argmax_d) * (h + 1 - argmax_h) * (w + 1 - argmax_w);
  if (d == argmax_d_low && h == argmax_h_low && w == argmax_w_high)
    weight = (d + 1 - argmax_d) * (h + 1 - argmax_h) * (argmax_w + 1 - w);
  if (d == argmax_d_low && h == argmax_h_high && w == argmax_w_low)
    weight = (d + 1 - argmax_d) * (argmax_h + 1 - h) * (w + 1 - argmax_w);
  if (d == argmax_d_low && h == argmax_h_high && w == argmax_w_high)
    weight = (d + 1 - argmax_d) * (argmax_h + 1 - h) * (argmax_w + 1 - w);
  if (d == argmax_d_high && h == argmax_h_low && w == argmax_w_low)
    weight = (argmax_d + 1 - d) * (h + 1 - argmax_h) * (w + 1 - argmax_w);
  if (d == argmax_d_high && h == argmax_h_low && w == argmax_w_high)
    weight = (argmax_d + 1 - d) * (h + 1 - argmax_h) * (argmax_w + 1 - w);
  if (d == argmax_d_high && h == argmax_h_high && w == argmax_w_low)
    weight = (argmax_d + 1 - d) * (argmax_h + 1 - h) * (w + 1 - argmax_w);
  if (d == argmax_d_high && h == argmax_h_high && w == argmax_w_high)
    weight = (argmax_d + 1 - d) * (argmax_h + 1 - h) * (argmax_w + 1 - w);
  return weight;
}

template <typename scalar_t>
__device__ scalar_t deformable_get_coordinate_weight(scalar_t argmax_d, scalar_t argmax_h, scalar_t argmax_w,
                                            const int depth, const int height, const int width, const scalar_t *im_data,
                                            const int data_height, const int data_width, const int bp_dir)
{
  if (argmax_d <= -1 || argmax_d >= depth || argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 || argmax_w >= width)
  {
    //empty
    return 0;
  }

  int argmax_d_low = floor(argmax_d);
  int argmax_h_low = floor(argmax_h);
  int argmax_w_low = floor(argmax_w);
  int argmax_d_high = argmax_d_low + 1;
  int argmax_h_high = argmax_h_low + 1;
  int argmax_w_high = argmax_w_low + 1;

  scalar_t weight = 0;

  if (bp_dir == 0)
  {
    if (argmax_d_low >= 0 && argmax_h_low >= 0 && argmax_w_low >= 0)
      weight += -1 * (argmax_h_low + 1 - argmax_h) * (argmax_w_low + 1 - argmax_w) * im_data[argmax_d_low * data_height * data_width + argmax_h_low * data_width + argmax_w_low];
    if (argmax_d_low >= 0 && argmax_h_low >= 0 && argmax_w_high <= width - 1)
      weight += -1 * (argmax_h_low + 1 - argmax_h) * (argmax_w - argmax_w_low) * im_data[argmax_d_low * data_height * data_width + argmax_h_low * data_width + argmax_w_high];
    if (argmax_d_low >= 0 && argmax_h_high <= height - 1 && argmax_w_low >= 0)
      weight += -1 * (argmax_h - argmax_h_low) * (argmax_w_low + 1 - argmax_w) * im_data[argmax_d_low * data_height * data_width + argmax_h_high * data_width + argmax_w_low];
    if (argmax_d_low >= 0 && argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
      weight += -1 * (argmax_h - argmax_h_low) * (argmax_w - argmax_w_low) * im_data[argmax_d_low * data_height * data_width + argmax_h_high * data_width + argmax_w_high];
    if (argmax_d_high <= depth - 1 && argmax_h_low >= 0 && argmax_w_low >= 0)
      weight += (argmax_h_low + 1 - argmax_h) * (argmax_w_low + 1 - argmax_w) * im_data[argmax_d_high * data_height * data_width + argmax_h_low * data_width + argmax_w_low];
    if (argmax_d_high <= depth - 1 && argmax_h_low >= 0 && argmax_w_high <= width - 1)
      weight += (argmax_h_low + 1 - argmax_h) * (argmax_w - argmax_w_low) * im_data[argmax_d_high * data_height * data_width + argmax_h_low * data_width + argmax_w_high];
    if (argmax_d_high <= depth - 1 && argmax_h_high <= height - 1 && argmax_w_low >= 0)
      weight += (argmax_h - argmax_h_low) * (argmax_w_low + 1 - argmax_w) * im_data[argmax_d_high * data_height * data_width + argmax_h_high * data_width + argmax_w_low];
    if (argmax_d_high <= depth - 1 && argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
      weight += (argmax_h - argmax_h_low) * (argmax_w - argmax_w_low) * im_data[argmax_d_high * data_height * data_width + argmax_h_high * data_width + argmax_w_high];
  }
  else if (bp_dir == 1)
  {
    if (argmax_d_low >= 0 && argmax_h_low >= 0 && argmax_w_low >= 0)
      weight += -1 * (argmax_d_low + 1 - argmax_d) * (argmax_w_low + 1 - argmax_w) * im_data[argmax_d_low * data_height * data_width + argmax_h_low * data_width + argmax_w_low];
    if (argmax_d_low >= 0 && argmax_h_low >= 0 && argmax_w_high <= width - 1)
      weight += -1 * (argmax_d_low + 1 - argmax_d) * (argmax_w - argmax_w_low) * im_data[argmax_d_low * data_height * data_width + argmax_h_low * data_width + argmax_w_high];
    if (argmax_d_low >= 0 && argmax_h_high <= height - 1 && argmax_w_low >= 0)
      weight += (argmax_d_low + 1 - argmax_d) * (argmax_w_low + 1 - argmax_w) * im_data[argmax_d_low * data_height * data_width + argmax_h_high * data_width + argmax_w_low];
    if (argmax_d_low >= 0 && argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
      weight += (argmax_d_low + 1 - argmax_d) * (argmax_w - argmax_w_low) * im_data[argmax_d_low * data_height * data_width + argmax_h_high * data_width + argmax_w_high];
    if (argmax_d_high <= depth - 1 && argmax_h_low >= 0 && argmax_w_low >= 0)
      weight += -1 * (argmax_d - argmax_d_low) * (argmax_w_low + 1 - argmax_w) * im_data[argmax_d_high * data_height * data_width + argmax_h_low * data_width + argmax_w_low];
    if (argmax_d_high <= depth - 1 && argmax_h_low >= 0 && argmax_w_high <= width - 1)
      weight += -1 * (argmax_d - argmax_d_low) * (argmax_w - argmax_w_low) * im_data[argmax_d_high * data_height * data_width + argmax_h_low * data_width + argmax_w_high];
    if (argmax_d_high <= depth - 1 && argmax_h_high <= height - 1 && argmax_w_low >= 0)
      weight += (argmax_d - argmax_d_low) * (argmax_w_low + 1 - argmax_w) * im_data[argmax_d_high * data_height * data_width + argmax_h_high * data_width + argmax_w_low];
    if (argmax_d_high <= depth - 1 && argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
      weight += (argmax_d - argmax_d_low) * (argmax_w - argmax_w_low) * im_data[argmax_d_high * data_height * data_width + argmax_h_high * data_width + argmax_w_high];
  }
  else if (bp_dir == 2)
  {
    if (argmax_d_low >= 0 && argmax_h_low >= 0 && argmax_w_low >= 0)
      weight += -1 * (argmax_d_low + 1 - argmax_d) * (argmax_h_low + 1 - argmax_h) * im_data[argmax_d_low * data_height * data_width + argmax_h_low * data_width + argmax_w_low];
    if (argmax_d_low >= 0 && argmax_h_low >= 0 && argmax_w_high <= width - 1)
      weight += (argmax_d_low + 1 - argmax_d) * (argmax_h_low + 1 - argmax_h) * im_data[argmax_d_low * data_height * data_width + argmax_h_low * data_width + argmax_w_high];
    if (argmax_d_low >= 0 && argmax_h_high <= height - 1 && argmax_w_low >= 0)
      weight += -1 * (argmax_d_low + 1 - argmax_d) * (argmax_h - argmax_h_low) * im_data[argmax_d_low * data_height * data_width + argmax_h_high * data_width + argmax_w_low];
    if (argmax_d_low >= 0 && argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
      weight += (argmax_d_low + 1 - argmax_d) * (argmax_h - argmax_h_low) * im_data[argmax_d_low * data_height * data_width + argmax_h_high * data_width + argmax_w_high];
    if (argmax_d_high <= depth - 1 && argmax_h_low >= 0 && argmax_w_low >= 0)
      weight += -1 * (argmax_d - argmax_d_low) * (argmax_h_low + 1 - argmax_h) * im_data[argmax_d_high * data_height * data_width + argmax_h_low * data_width + argmax_w_low];
    if (argmax_d_high <= depth - 1 && argmax_h_low >= 0 && argmax_w_high <= width - 1)
      weight += (argmax_d - argmax_d_low) * (argmax_h_low + 1 - argmax_h) * im_data[argmax_d_high * data_height * data_width + argmax_h_low * data_width + argmax_w_high];
    if (argmax_d_high <= depth - 1 && argmax_h_high <= height - 1 && argmax_w_low >= 0)
      weight += -1 * (argmax_d - argmax_d_low) * (argmax_h - argmax_h_low) * im_data[argmax_d_high * data_height * data_width + argmax_h_high * data_width + argmax_w_low];
    if (argmax_d_high <= depth - 1 && argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
      weight += (argmax_d - argmax_d_low) * (argmax_h - argmax_h_low) * im_data[argmax_d_high * data_height * data_width + argmax_h_high * data_width + argmax_w_high];
  }

  return weight;
}

template <typename scalar_t>
__global__ void deformable_im2col_gpu_kernel(const int n,
                                                       const scalar_t *data_im, const scalar_t *data_offset,
                                                       const int depth, const int height, const int width, const int kernel_d, const int kernel_h, const int kernel_w,
                                                       const int pad_d, const int pad_h, const int pad_w,
                                                       const int stride_d, const int stride_h, const int stride_w,
                                                       const int dilation_d, const int dilation_h, const int dilation_w,
                                                       const int channel_per_deformable_group,
                                                       const int batch_size, const int num_channels, const int deformable_group,
                                                       const int depth_col, const int height_col, const int width_col,
                                                       scalar_t *data_col)
{
  // launch channels * batch_size * depth_col * height_col * width_col cores
  CUDA_KERNEL_LOOP(index, n)
  {
    // NOTE(CharlesShang): different from Dai Jifeng's MXNet implementation, col_buffer is of shape (c*kd*kw*kh, N, od, oh, ow)
    // here columns is of shape (N, c*kd*kw*kh, od * oh * ow), need to adapt axis
    // NOTE(Jiarui XU): different from CharlesShang's implementation, col_buffer is of shape (N, c*kd*kw*kh, od * oh * ow)
    // here columns is of shape (c*kd*kw*kh, N, od * oh, ow), need to adapt axis

    // index index of output matrix

    const int w_col = index % width_col;
    const int h_col = (index / width_col) % height_col;
    const int d_col = (index / width_col / height_col) % depth_col;
    const int b_col = (index / width_col / height_col / depth_col) % batch_size;
    const int c_im = (index / width_col / height_col / depth_col) / batch_size;
    const int c_col = c_im * kernel_d * kernel_h * kernel_w;

    // compute deformable group index
    const int deformable_group_index = c_im / channel_per_deformable_group;

    const int d_in = d_col * stride_d - pad_d;
    const int h_in = h_col * stride_h - pad_h;
    const int w_in = w_col * stride_w - pad_w;

     scalar_t *data_col_ptr = data_col + (((c_col * batch_size + b_col) * depth_col + d_col) * height_col + h_col) * width_col + w_col;
    // const scalar_t* data_im_ptr = data_im + (((b_col * num_channels + c_im) * depth + d_in) * height + h_in) * width + w_in;
    const scalar_t *data_im_ptr = data_im + (b_col * num_channels + c_im) * depth * height * width;
    const scalar_t *data_offset_ptr = data_offset + (b_col * deformable_group + deformable_group_index) * 3 * kernel_d * kernel_h * kernel_w * depth_col * height_col * width_col;
    for (int i = 0; i < kernel_d; ++i)
    {
      for (int j = 0; j < kernel_h; ++j)
      {
        for (int k = 0; k < kernel_w; ++k)
        {
          const int data_offset_d_ptr = ((3 * (i * kernel_h * kernel_w + j * kernel_w + k) * depth_col + d_col) * height_col + h_col) * width_col + w_col;
          const int data_offset_h_ptr = (((3 * (i * kernel_h * kernel_w + j * kernel_w + k) + 1) * depth_col + d_col) * height_col + h_col) * width_col + w_col;
          const int data_offset_w_ptr = (((3 * (i * kernel_h * kernel_w + j * kernel_w + k) + 2) * depth_col + d_col) * height_col + h_col) * width_col + w_col;
          const scalar_t offset_d = data_offset_ptr[data_offset_d_ptr];
          const scalar_t offset_h = data_offset_ptr[data_offset_h_ptr];
          const scalar_t offset_w = data_offset_ptr[data_offset_w_ptr];
          scalar_t val = static_cast<scalar_t>(0);
          const scalar_t d_im = d_in + i * dilation_d + offset_d;
          const scalar_t h_im = h_in + j * dilation_h + offset_h;
          const scalar_t w_im = w_in + k * dilation_w + offset_w;
          if (d_im > -1 && h_im > -1 && w_im > -1 && d_im < depth && h_im < height && w_im < width)
          {
            //const scalar_t map_d = i * dilation_d + offset_d;
            //const scalar_t map_h = j * dilation_h + offset_h;
            //const scalar_t map_w = k * dilation_w + offset_w;
            //const int cur_depth = depth - d_in;
            //const int cur_height = height - h_in;
            //const int cur_width = width - w_in;
            //val = deformable_im2col_bilinear(data_im_ptr, height, width, cur_depth, cur_height, cur_width, map_d, map_h, map_w);
            val = deformable_im2col_bilinear(data_im_ptr, height, width, depth, height, width, d_im, h_im, w_im);
          }
          *data_col_ptr = val;
          data_col_ptr += batch_size * depth_col * height_col * width_col;
        }
      }
    }
  }
}

template <typename scalar_t>
__global__ void deformable_col2im_gpu_kernel(
    const int n,
    const scalar_t *data_col, const scalar_t *data_offset,
    const int channels, const int depth, const int height, const int width,
    const int kernel_d, const int kernel_h, const int kernel_w,
    const int pad_d, const int pad_h, const int pad_w,
    const int stride_d, const int stride_h, const int stride_w,
    const int dilation_d, const int dilation_h, const int dilation_w,
    const int channel_per_deformable_group,
    const int batch_size, const int deformable_group,
    const int depth_col, const int height_col, const int width_col,
    scalar_t *grad_im)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    const int k = (index / depth_col / width_col / height_col / batch_size) % kernel_w;
    const int j = (index / depth_col / width_col / height_col / batch_size / kernel_w) % kernel_h;
    const int i = (index / depth_col / width_col / height_col / batch_size / kernel_w / kernel_h) % kernel_d;
    const int c = index / depth_col / width_col / height_col / batch_size / kernel_w / kernel_h / kernel_d;
    // compute the start and end of the output

    const int deformable_group_index = c / channel_per_deformable_group;

    int w_out = index % width_col;
    int h_out = (index / width_col) % height_col;
    int d_out = (index / width_col / height_col) % depth_col;
    int b = (index / width_col / height_col / depth_col) % batch_size;
    int d_in = d_out * stride_d - pad_d;
    int w_in = w_out * stride_w - pad_w;
    int h_in = h_out * stride_h - pad_h;

    const scalar_t *data_offset_ptr = data_offset + (b * deformable_group + deformable_group_index) * 3 * kernel_d * kernel_h * kernel_w * depth_col * height_col * width_col;
    const int data_offset_d_ptr = (((3 * (i * kernel_h * kernel_w + j * kernel_w + k)) * depth_col + d_out) * height_col + h_out) * width_col + w_out;
    const int data_offset_h_ptr = (((3 * (i * kernel_h * kernel_w + j * kernel_w + k) + 1) * depth_col + d_out) * height_col + h_out) * width_col + w_out;
    const int data_offset_w_ptr = (((3 * (i * kernel_h * kernel_w + j * kernel_w + k) + 2) * depth_col + d_out) * height_col + h_out) * width_col + w_out;
    const scalar_t offset_d = data_offset_ptr[data_offset_d_ptr];
    const scalar_t offset_h = data_offset_ptr[data_offset_h_ptr];
    const scalar_t offset_w = data_offset_ptr[data_offset_w_ptr];
    const scalar_t cur_inv_d_data = d_in + i * dilation_d + offset_d;
    const scalar_t cur_inv_h_data = h_in + j * dilation_h + offset_h;
    const scalar_t cur_inv_w_data = w_in + k * dilation_w + offset_w;

    const scalar_t cur_top_grad = data_col[index];
    const int cur_d = (int)cur_inv_d_data;
    const int cur_h = (int)cur_inv_h_data;
    const int cur_w = (int)cur_inv_w_data;
    for (int dz = -2; dz <= 2; dz++)
    {
      for (int dy = -2; dy <= 2; dy++)
      {
        for (int dx = -2; dx <= 2; dx++)
        {
          if (cur_d + dz >= 0 && cur_d + dz < depth &&
              cur_h + dy >= 0 && cur_h + dy < height &&
              cur_w + dx >= 0 && cur_w + dx < width &&
              abs(cur_inv_d_data - (cur_d + dz)) < 1 &&
              abs(cur_inv_h_data - (cur_h + dy)) < 1 &&
              abs(cur_inv_w_data - (cur_w + dx)) < 1)
          {
            int cur_bottom_grad_pos = (((b * channels + c) * depth + cur_d + dz) * height + cur_h + dy) * width + cur_w + dx;
            scalar_t weight = deformable_get_gradient_weight(cur_inv_d_data, cur_inv_h_data, cur_inv_w_data, cur_d + dz, cur_h + dy, cur_w + dx, depth, height, width);
            atomicAdd(grad_im + cur_bottom_grad_pos, weight * cur_top_grad);
          }
        }
      }
    }
  }
}

template <typename scalar_t>
__global__ void deformable_col2im_coord_gpu_kernel(const int n,
                                                   const scalar_t *data_col, const scalar_t *data_im,
                                                   const scalar_t *data_offset,
                                                   const int channels, const int depth, const int height, const int width,
                                                   const int kernel_d, const int kernel_h, const int kernel_w,
                                                   const int pad_d, const int pad_h, const int pad_w,
                                                   const int stride_d, const int stride_h, const int stride_w,
                                                   const int dilation_d, const int dilation_h, const int dilation_w,
                                                   const int channel_per_deformable_group,
                                                   const int batch_size, const int offset_channels, const int deformable_group,
                                                   const int depth_col, const int height_col, const int width_col,
                                                   scalar_t *grad_offset)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    scalar_t val = 0;
    int w = index % width_col;
    int h = (index / width_col) % height_col;
    int d = (index / width_col / height_col) % depth_col;
    int c = (index / width_col / height_col / depth_col) % offset_channels;
    int b = (index / width_col / height_col / depth_col) / offset_channels;
    // compute the start and end of the output

    const int deformable_group_index = c / (3 * kernel_d * kernel_h * kernel_w);
    const int col_step = kernel_d * kernel_h * kernel_w;
    int cnt = 0;
    const scalar_t *data_col_ptr = data_col + deformable_group_index * channel_per_deformable_group * batch_size * depth_col * width_col * height_col;
    const scalar_t *data_im_ptr = data_im + (b * deformable_group + deformable_group_index) * channel_per_deformable_group / kernel_d / kernel_h / kernel_w * depth * height * width;
    const scalar_t *data_offset_ptr = data_offset + (b * deformable_group + deformable_group_index) * 3 * kernel_d * kernel_h * kernel_w * depth_col * height_col * width_col;

    const int offset_c = c - deformable_group_index * 3 * kernel_d * kernel_h * kernel_w;

    for (int col_c = (offset_c / 3); col_c < channel_per_deformable_group; col_c += col_step)
    {
      const int col_pos = (((col_c * batch_size + b) *depth_col + d) * height_col + h) * width_col + w;
      const int bp_dir = offset_c % 3;

      int k = (col_pos / width_col / height_col / depth_col / batch_size) % kernel_w;
      int j = (col_pos / width_col / height_col / depth_col / batch_size / kernel_w) % kernel_h;
      int i = (col_pos / width_col / height_col / depth_col / batch_size / kernel_w / kernel_h) % kernel_d;
      int w_out = col_pos % width_col;
      int h_out = (col_pos / width_col) % height_col;
      int d_out = (col_pos / width_col / height_col) % depth_col;
      int d_in = d_out * stride_d - pad_d;
      int w_in = w_out * stride_w - pad_w;
      int h_in = h_out * stride_h - pad_h;
      const int data_offset_d_ptr = ((3 * (i * kernel_h * kernel_w + j * kernel_w + k) * depth_col + d_out) * height_col + h_out) * width_col + w_out;
      const int data_offset_h_ptr = (((3 * (i * kernel_h * kernel_w + j * kernel_w + k) + 1) * depth_col + d_out) * height_col + h_out) * width_col + w_out;
      const int data_offset_w_ptr = (((3 * (i * kernel_h * kernel_w + j * kernel_w + k) + 2) * depth_col + d_out) * height_col + h_out) * width_col + w_out;
      const scalar_t offset_d = data_offset_ptr[data_offset_d_ptr];
      const scalar_t offset_h = data_offset_ptr[data_offset_h_ptr];
      const scalar_t offset_w = data_offset_ptr[data_offset_w_ptr];
      scalar_t inv_d = d_in + i * dilation_d + offset_d;
      scalar_t inv_h = h_in + j * dilation_h + offset_h;
      scalar_t inv_w = w_in + k * dilation_w + offset_w;
      if (inv_d <= -1 || inv_h <= -1 || inv_w <= -1 || inv_d >= depth || inv_h >= height || inv_w >= width)
      {
        inv_d = inv_h = inv_w = -2;
      }
      const scalar_t weight = deformable_get_coordinate_weight(
          inv_d, inv_h, inv_w,
          depth, height, width, data_im_ptr + cnt * depth * height * width, height, width, bp_dir);
      val += weight * data_col_ptr[col_pos];
      cnt += 1;
    }
    // KERNEL_ASSIGN(grad_offset[index], offset_req, val);
    grad_offset[index] = val;
  }
}

template <typename scalar_t>
void deformable_im2col_cuda(cudaStream_t stream,
  const scalar_t* data_im, const scalar_t* data_offset,
  const int batch_size, const int channels, const int depth_im, const int height_im, const int width_im,
  const int depth_col, const int height_col, const int width_col, const int kernel_d, const int kernel_h, const int kernel_w,
  const int pad_d, const int pad_h, const int pad_w, const int stride_d, const int stride_h, const int stride_w,
  const int dilation_d, const int dilation_h, const int dilation_w,
  const int deformable_group, scalar_t* data_col) {
  // num_axes should be smaller than block size
  const int channel_per_deformable_group = channels / deformable_group;
  const int num_kernels = channels * batch_size * depth_col * height_col * width_col;
  deformable_im2col_gpu_kernel<scalar_t>
      <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS,
          0, stream>>>(
      num_kernels, data_im, data_offset, depth_im, height_im, width_im, kernel_d, kernel_h, kernel_w,
      pad_d, pad_h, pad_w, stride_d, stride_h, stride_w, dilation_d, dilation_h, dilation_w, channel_per_deformable_group,
      batch_size, channels, deformable_group, depth_col, height_col, width_col, data_col);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in deformable_im2col_cuda: %s\n", cudaGetErrorString(err));
  }

}

template <typename scalar_t>
void deformable_col2im_cuda(cudaStream_t stream,
  const scalar_t* data_col, const scalar_t* data_offset,
  const int batch_size, const int channels, const int depth_im, const int height_im, const int width_im,
  const int depth_col, const int height_col, const int width_col, const int kernel_d, const int kernel_h, const int kernel_w,
  const int pad_d, const int pad_h, const int pad_w, const int stride_d, const int stride_h, const int stride_w,
  const int dilation_d, const int dilation_h, const int dilation_w,
  const int deformable_group, scalar_t* grad_im){

  const int channel_per_deformable_group = channels / deformable_group;
  const int num_kernels = channels * kernel_d * kernel_h * kernel_w * batch_size * depth_col * height_col * width_col;
  deformable_col2im_gpu_kernel<scalar_t>
      <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS,
          0, stream>>>(
        num_kernels, data_col, data_offset, channels, depth_im, height_im, width_im,
        kernel_d, kernel_h, kernel_w, pad_d, pad_h, pad_h, stride_d, stride_h, stride_w,
        dilation_d, dilation_h, dilation_w, channel_per_deformable_group,
        batch_size, deformable_group, depth_col, height_col, width_col, grad_im);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in deformable_col2im_cuda: %s\n", cudaGetErrorString(err));
  }

}

template <typename scalar_t>
void deformable_col2im_coord_cuda(cudaStream_t stream,
  const scalar_t* data_col, const scalar_t* data_im, const scalar_t* data_offset,
  const int batch_size, const int channels, const int depth_im, const int height_im, const int width_im,
  const int depth_col, const int height_col, const int width_col, const int kernel_d, const int kernel_h, const int kernel_w,
  const int pad_d, const int pad_h, const int pad_w, const int stride_d, const int stride_h, const int stride_w,
  const int dilation_d, const int dilation_h, const int dilation_w,
  const int deformable_group,
  scalar_t* grad_offset) {
  const int num_kernels = batch_size * depth_col * height_col * width_col * 3 * kernel_d * kernel_h * kernel_w * deformable_group;
  const int channel_per_deformable_group = channels * kernel_d * kernel_h * kernel_w / deformable_group;
  deformable_col2im_coord_gpu_kernel<scalar_t>
      <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS,
        0, stream>>>(
        num_kernels, data_col, data_im, data_offset, channels, depth_im, height_im, width_im,
        kernel_d, kernel_h, kernel_w, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w,
        dilation_d, dilation_h, dilation_w, channel_per_deformable_group,
        batch_size, 3 * kernel_d * kernel_h * kernel_w * deformable_group, deformable_group, depth_col, height_col, width_col,
        grad_offset);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in deformable_col2im_coord_cuda: %s\n", cudaGetErrorString(err));
  }
}


at::Tensor
deform_conv_cuda_forward(const at::Tensor &input,
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
                    const int im2col_step)
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
            deformable_im2col_cuda(at::cuda::getCurrentCUDAStream(),
                                             input.data<scalar_t>() + n * im2col_step_ * per_input_size,
                                             offset.data<scalar_t>() + n * im2col_step_ * per_offset_size,
                                             batch_n, channels, depth, height, width,
                                             depth_out, height_out, width_out, kernel_d, kernel_h, kernel_w,
                                             pad_d, pad_h, pad_w, stride_d, stride_h, stride_w, dilation_d, dilation_h, dilation_w,
                                             deformable_group,
                                             columns.data<scalar_t>());

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

std::vector<at::Tensor> deform_conv_cuda_backward(const at::Tensor &input,
                                             const at::Tensor &weight,
                                             const at::Tensor &bias,
                                             const at::Tensor &offset,
                                             const at::Tensor &grad_output,
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
                                             const int im2col_step)
{

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

    const int batch_ = grad_output.size(0);
    const int channels_out_ = grad_output.size(1);
    const int depth_out_ = grad_output.size(2);
    const int height_out_ = grad_output.size(3);
    const int width_out_ = grad_output.size(4);

    const int im2col_step_ = std::min(im2col_step, batch);

    AT_ASSERTM(batch % im2col_step_ == 0, "batch(%d) must divide im2col_step(%d)", batch, im2col_step_);

    AT_ASSERTM((channels % group == 0) && (channels_out % group == 0), 
        "channels(%d) and channels_out(%d) must divide group(%d)", channels, channels_out, group);

    AT_ASSERTM(kernel_h_ == kernel_h && kernel_w_ == kernel_w && kernel_d_ == kernel_d,
               "Input shape and kernel shape wont match: (%d x %d x %d vs %d x %d x %d).", kernel_d, kernel_h, kernel_w, kernel_d_, kernel_h_, kernel_w_);

    AT_ASSERTM(channels == (channels_kernel * group),
               "Input shape and kernel channels wont match: (%d vs %d).", channels, channels_kernel * group);

    const int depth_out = (depth + 2 * pad_d - (dilation_d * (kernel_d - 1) + 1)) / stride_d + 1;
    const int height_out = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int width_out = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

    AT_ASSERTM(batch == batch_,
               "Input shape and grad_out batch wont match: (%d vs %d).", batch, batch_);

    AT_ASSERTM(channels_out == channels_out_,
               "Input shape and grad_out channels_out wont match: (%d vs %d).", channels_out, channels_out_);

    AT_ASSERTM(height_out == height_out_ && width_out == width_out_ && depth_out == depth_out_,
               "Input shape and grad_out shape wont match: (%d x %d x %d vs %d x %d x %d).", depth_out, depth_out_, height_out, height_out_, width_out, width_out_);

    auto grad_input = at::zeros_like(input);
    auto grad_offset = at::zeros_like(offset);
    auto grad_weight = at::zeros_like(weight);
    auto grad_bias = at::zeros_like(bias);

    // auto grad_output_m = grad_output.permute({1, 0, 2, 3, 4}).contiguous().view({channels_out, batch * depth_out * height_out * width_out});
    // auto weight_m = weight.view({channels_out, channels_kernel * kernel_d * kernel_h * kernel_w}).t();
    // columns = at::mm(weight_m, grad_output_m);

    // prepare group weight and bias
    auto weight_g = weight.view({group, channels_out/group, channels_kernel, kernel_d, kernel_h, kernel_w});
    auto grad_weight_g = grad_weight.view({group, channels_out/group, channels_kernel, kernel_d, kernel_h, kernel_w});
    auto grad_bias_g = grad_bias.view({group, channels_out/group});

    const int batch_n = im2col_step_;
    const int per_input_size = channels * depth * height * width;
    const int per_offset_size = offset.size(1) * offset.size(2) * offset.size(3) * offset.size(4);
    auto grad_output_n = grad_output.view({batch/im2col_step_, batch_n, channels_out, depth_out, height_out, width_out});
    for (int n = 0; n < batch/im2col_step_; ++n)
    {
        auto grad_output_g = grad_output_n.select(0, n).view({batch_n, group, channels_out/group, depth_out, height_out, width_out});
        auto ones = at::ones({batch_n * depth_out * height_out * width_out}, input.options());
        auto columns = at::empty({channels * kernel_d * kernel_h * kernel_w, batch_n * 1 * depth_out * height_out * width_out}, input.options());
        auto columns_g = columns.view({group, channels/group * kernel_d * kernel_h * kernel_w, batch_n * depth_out * height_out * width_out});
        for (int g = 0; g < group; ++g)
        {
            auto grad_output_gm = grad_output_g.select(1, g).permute({1, 0, 2, 3, 4}).contiguous().view({channels_out/group, batch_n * depth_out * height_out * width_out});
            auto weight_gm = weight_g.select(0, g).view({channels_out/group, channels_kernel * kernel_d * kernel_h * kernel_w}).t();
            columns_g.select(0, g) = at::mm(weight_gm, grad_output_gm);
        }

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "deform_conv_backward_cuda", ([&] {
            deformable_col2im_coord_cuda(at::cuda::getCurrentCUDAStream(),
                                                   columns.data<scalar_t>(),
                                                   input.data<scalar_t>() + n * im2col_step_ * per_input_size,
                                                   offset.data<scalar_t>() + n * im2col_step_ * per_offset_size,
                                                   batch_n, channels, depth, height, width,
                                                   depth_out, height_out, width_out, kernel_d, kernel_h, kernel_w,
                                                   pad_d, pad_h, pad_w, stride_d, stride_h, stride_w,
                                                   dilation_d, dilation_h, dilation_w, deformable_group,
                                                   grad_offset.data<scalar_t>() + n * im2col_step_ * per_offset_size);
            // gradient w.r.t. input data
            deformable_col2im_cuda(at::cuda::getCurrentCUDAStream(),
                                             columns.data<scalar_t>(),
                                             offset.data<scalar_t>() + n * im2col_step_ * per_offset_size,
                                             batch_n, channels, depth, height, width,
                                             depth_out, height_out, width_out, kernel_d, kernel_h, kernel_w,
                                             pad_d, pad_h, pad_w, stride_d, stride_h, stride_w,
                                             dilation_d, dilation_h, dilation_w, deformable_group,
                                             grad_input.data<scalar_t>() + n * im2col_step_ * per_input_size);

            // gradient w.r.t. weight, dWeight should accumulate across the batch and group
            deformable_im2col_cuda(at::cuda::getCurrentCUDAStream(),
                                             input.data<scalar_t>() + n * im2col_step_ * per_input_size,
                                             offset.data<scalar_t>() + n * im2col_step_ * per_offset_size,
                                             batch_n, channels, depth, height, width,
                                             depth_out, height_out, width_out, kernel_d, kernel_h, kernel_w,
                                             pad_d, pad_h, pad_w, stride_d, stride_h, stride_w,
                                             dilation_d, dilation_h, dilation_w, deformable_group,
                                             columns.data<scalar_t>());

        }));

        // auto grad_output_m = grad_output.permute({1, 0, 2, 3, 4}).contiguous().view({channels_out, batch * depth_out * height_out * width_out});
        // grad_weight = at::mm(grad_output_m, columns.t()).view_as(weight);
        // grad_bias = at::mv(grad_output_m, ones);
        // auto grad_output_g = grad_output.view({batch, group, channels_out/group, depth_out, height_out, width_out});
        // auto columns_g = columns.view({group, channels/group * kernel_d * kernel_h * kernel_w, batch * depth_out * height_out * width_out});
        for (int g = 0; g < group; ++g)
        {
            auto grad_output_gm = grad_output_g.select(1, g).permute({1, 0, 2, 3, 4}).contiguous().view({channels_out/group, batch_n * depth_out * height_out * width_out});
            auto columns_gm = columns_g.select(0, g).t();
            auto grad_weight_gm = grad_weight_g.select(0, g).view({channels_out/group, channels_kernel * kernel_d * kernel_h * kernel_w});
            auto grad_bias_gm = grad_bias_g.select(0, g);
            grad_weight_g.select(0, g) = at::addmm(grad_weight_gm, grad_output_gm, columns_gm).view_as(grad_weight_g.select(0, g));
            grad_bias_g.select(0, g) = at::addmv(grad_bias_gm, grad_output_gm, ones);
        }

    }

    return {
        grad_input, grad_offset, grad_weight, grad_bias
    };
}
