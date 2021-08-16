#include <stdio.h>
#include <math.h>
#include <float.h>

#include "cuda_helpers.h"
#include "interpolation_helpers_3d.h"

template <typename scalar_t>
__global__ void deformable_im2col_gpu_kernel(
    const int n,
    const scalar_t *data_im, const scalar_t *data_offset,
    const int depth, const int height, const int width,
    const int kernel_d, const int kernel_h, const int kernel_w,
    const int pad_d, const int pad_h, const int pad_w,
    const int stride_d, const int stride_h, const int stride_w,
    const int dilation_d, const int dilation_h, const int dilation_w,
    const int channel_per_deformable_group,
    const int batch_size, const int num_channels, const int deformable_group,
    const int depth_col, const int height_col, const int width_col,
    scalar_t *data_col, const int order)
{
  // launch channels * batch_size * depth_col * height_col * width_col cores
  CUDA_1D_KERNEL_LOOP(index, n)
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
            if (order == 0) {
              val = nearest_3d_interpolate(data_im_ptr, depth, height, width, d_im, h_im, w_im, index);
            }else if (order == 1) {
              val = linear_3d_interpolate(data_im_ptr, depth, height, width, d_im, h_im, w_im, index);
            }else if (order == 3) {
              val = hermite_3d_interpolate(data_im_ptr, depth, height, width, d_im, h_im, w_im, index);
            }
          }
          *data_col_ptr = val;
          data_col_ptr += batch_size * depth_col * height_col * width_col;
        }
      }
    }
  }
}