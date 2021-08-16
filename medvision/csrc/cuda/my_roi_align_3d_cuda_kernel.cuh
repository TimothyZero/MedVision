#ifndef ROI_ALIGN_CUDA_KERNEL_CUH
#define ROI_ALIGN_CUDA_KERNEL_CUH
#include <stdio.h>
#include <float.h>

#include "cuda_helpers.h"
#include "interpolation_helpers_3d.h"

/*** Forward ***/
template <typename scalar_t>
__global__ void roi_align_3d_forward_cuda_kernel(
    const int nthreads, const scalar_t *bottom_data,
    const scalar_t *bottom_rois, const scalar_t spatial_scale,
    const int sampling_ratio, const int order,
    const int channels,
    const int depth, const int height, const int width,
    const int pooled_depth, const int pooled_height, const int pooled_width,
    scalar_t *top_data) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, pd, ph, pw) is an element in the pooled output
    int pw =  index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int pd = (index / pooled_width / pooled_height) % pooled_depth;
    int c  = (index / pooled_width / pooled_height / pooled_depth) % channels;
    int n  =  index / pooled_width / pooled_height / pooled_depth / channels;

    // batch_idx, x1, y1, z1, x2, y2, z2
    const scalar_t *offset_bottom_rois = bottom_rois + n * 7;
    int roi_batch_ind = offset_bottom_rois[0];

    // Do not using rounding; this implementation detail is critical
    scalar_t roi_start_w = offset_bottom_rois[1] * spatial_scale;
    scalar_t roi_start_h = offset_bottom_rois[2] * spatial_scale;
    scalar_t roi_start_d = offset_bottom_rois[3] * spatial_scale;
    scalar_t roi_end_w = offset_bottom_rois[4] * spatial_scale;
    scalar_t roi_end_h = offset_bottom_rois[5] * spatial_scale;
    scalar_t roi_end_d = offset_bottom_rois[6] * spatial_scale;
    scalar_t roi_width = max(roi_end_w - roi_start_w, (scalar_t)1.);
    scalar_t roi_height = max(roi_end_h - roi_start_h, (scalar_t)1.);
    scalar_t roi_depth = max(roi_end_d - roi_start_d, (scalar_t)1.);

    roi_width = max(roi_width, (scalar_t)1.);
    roi_height = max(roi_height, (scalar_t)1.);
    roi_depth = max(roi_depth, (scalar_t)1.);

    scalar_t bin_size_d = static_cast<scalar_t>(roi_depth) / static_cast<scalar_t>(pooled_depth);
    scalar_t bin_size_h = static_cast<scalar_t>(roi_height) / static_cast<scalar_t>(pooled_height);
    scalar_t bin_size_w = static_cast<scalar_t>(roi_width) / static_cast<scalar_t>(pooled_width);

    const scalar_t *offset_bottom_data = bottom_data + (roi_batch_ind * channels + c) * depth * height * width;

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_d = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_depth / pooled_depth);
    int roi_bin_grid_h = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_height / pooled_height);  // e.g., = 2
    int roi_bin_grid_w = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

    // We do average (integral) pooling inside a bin
    const scalar_t count = max(roi_bin_grid_d * roi_bin_grid_h * roi_bin_grid_w, 1);  // e.g. = 4

    scalar_t output_val = 0.;
    for (int iz = 0; iz < roi_bin_grid_d; iz++)
    {
      const scalar_t z = roi_start_d + pd * bin_size_d + static_cast<scalar_t>(iz + .5f) * bin_size_d / static_cast<scalar_t>(roi_bin_grid_d);
      for (int iy = 0; iy < roi_bin_grid_h; iy++) // e.g., iy = 0, 1
      {
        const scalar_t y = roi_start_h + ph * bin_size_h + static_cast<scalar_t>(iy + .5f) * bin_size_h / static_cast<scalar_t>(roi_bin_grid_h); // e.g., 0.5, 1.5, always in the middle of two grid pointsk
        for (int ix = 0; ix < roi_bin_grid_w; ix++)
        {
          const scalar_t x = roi_start_w + pw * bin_size_w + static_cast<scalar_t>(ix + .5f) * bin_size_w / static_cast<scalar_t>(roi_bin_grid_w);

          scalar_t val;
          if (order == 0) {
            val = nearest_3d_interpolate<scalar_t>(offset_bottom_data, depth, height, width, z, y, x, index);
          }
          else if (order == 1) {
            val = linear_3d_interpolate<scalar_t>(offset_bottom_data, depth, height, width, z, y, x, index);
          }
          output_val += val;
        }
      }
    }
    output_val /= count;

    top_data[index] = output_val;
  }
}

/*** Backward ***/
template <typename scalar_t>
__global__ void roi_align_3d_backward_cuda_kernel(
    const int nthreads, const scalar_t *top_diff, const scalar_t *bottom_rois,
    const scalar_t spatial_scale, const int sampling_ratio, const int order,
    const int channels,
    const int depth, const int height, const int width,
    const int pooled_depth, const int pooled_height, const int pooled_width,
    scalar_t *bottom_diff) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, pd, ph, pw) is an element in the pooled output
    int pw =  index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int pd = (index / pooled_width / pooled_height) % pooled_depth;
    int c  = (index / pooled_width / pooled_height / pooled_depth) % channels;
    int n  =  index / pooled_width / pooled_height / pooled_depth / channels;

    // batch_idx, ctr_x, ctr_y, ctr_z, w, h, d, angle1, angle2, angle3
    const scalar_t *offset_bottom_rois = bottom_rois + n * 7;
    int roi_batch_ind = offset_bottom_rois[0];

    // Do not round
    scalar_t roi_start_w = offset_bottom_rois[1] * spatial_scale;
    scalar_t roi_start_h = offset_bottom_rois[2] * spatial_scale;
    scalar_t roi_start_d = offset_bottom_rois[3] * spatial_scale;
    scalar_t roi_end_w = offset_bottom_rois[4] * spatial_scale;
    scalar_t roi_end_h = offset_bottom_rois[5] * spatial_scale;
    scalar_t roi_end_d = offset_bottom_rois[6] * spatial_scale;
    scalar_t roi_width = max(roi_end_w - roi_start_w, (scalar_t)1.);
    scalar_t roi_height = max(roi_end_h - roi_start_h, (scalar_t)1.);
    scalar_t roi_depth = max(roi_end_d - roi_start_d, (scalar_t)1.);

    roi_width = max(roi_width, (scalar_t)1.);
    roi_height = max(roi_height, (scalar_t)1.);
    roi_depth = max(roi_depth, (scalar_t)1.);

    scalar_t bin_size_d = static_cast<scalar_t>(roi_depth) / static_cast<scalar_t>(pooled_depth);
    scalar_t bin_size_h = static_cast<scalar_t>(roi_height) / static_cast<scalar_t>(pooled_height);
    scalar_t bin_size_w = static_cast<scalar_t>(roi_width) / static_cast<scalar_t>(pooled_width);

    scalar_t *offset_bottom_diff =
        bottom_diff + (roi_batch_ind * channels + c) * depth * height * width;

    int top_offset = (n * channels + c) * pooled_depth * pooled_height * pooled_width;
    const scalar_t *offset_top_diff = top_diff + top_offset;
    const scalar_t top_diff_this_bin = offset_top_diff[pd * pooled_height * pooled_width + ph * pooled_width + pw];

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_d = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_depth / pooled_depth);
    int roi_bin_grid_h = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_height / pooled_height);  // e.g., = 2
    int roi_bin_grid_w = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

    // We do average (integral) pooling inside a bin
    const scalar_t count = max(roi_bin_grid_d * roi_bin_grid_h * roi_bin_grid_w, 1);  // e.g. = 4

    for (int iz = 0; iz < roi_bin_grid_d; iz++)
    {
      const scalar_t z = roi_start_d + pd * bin_size_d +
          static_cast<scalar_t>(iz + .5f) * bin_size_d / static_cast<scalar_t>(roi_bin_grid_d);

      for (int iy = 0; iy < roi_bin_grid_h; iy++) // e.g., iy = 0, 1
      {
        const scalar_t y = roi_start_h + ph * bin_size_h +
            static_cast<scalar_t>(iy + .5f) * bin_size_h / static_cast<scalar_t>(roi_bin_grid_h); // e.g., 0.5, 1.5, always in the middle of two grid pointsk

        for (int ix = 0; ix < roi_bin_grid_w; ix++)
        {
          const scalar_t x = roi_start_w + pw * bin_size_w +
              static_cast<scalar_t>(ix + .5f) * bin_size_w / static_cast<scalar_t>(roi_bin_grid_w);

          scalar_t w1, w2, w3, w4, w5, w6, w7, w8;
          int x0, x1, y0, y1, z0, z1;
          if (order == 0) {
            nearest_3d_interpolate_gradient<scalar_t>(depth, height, width,
                                                      z, y, x,
                                                      w1, w2, w3, w4, w5, w6, w7, w8,
                                                      x0, x1,
                                                      y0, y1,
                                                      z0, z1,
                                                      index);
          }
          else if (order == 1) {
            linear_3d_interpolate_gradient<scalar_t>(depth, height, width,
                                                     z, y, x,
                                                     w1, w2, w3, w4, w5, w6, w7, w8,
                                                     x0, x1,
                                                     y0, y1,
                                                     z0, z1,
                                                     index);
          }

          scalar_t g1 = top_diff_this_bin * w1 / count;
          scalar_t g2 = top_diff_this_bin * w2 / count;
          scalar_t g3 = top_diff_this_bin * w3 / count;
          scalar_t g4 = top_diff_this_bin * w4 / count;
          scalar_t g5 = top_diff_this_bin * w5 / count;
          scalar_t g6 = top_diff_this_bin * w6 / count;
          scalar_t g7 = top_diff_this_bin * w7 / count;
          scalar_t g8 = top_diff_this_bin * w8 / count;

          if (x0 >= 0 && x1 >= 0 && y0 >= 0 && y1 >= 0 && z0 >= 0 && z1 >= 0) {
            atomicAdd(offset_bottom_diff + z0 * height * width + y0 * width + x0, g1);
            atomicAdd(offset_bottom_diff + z0 * height * width + y0 * width + x1, g2);
            atomicAdd(offset_bottom_diff + z0 * height * width + y1 * width + x0, g3);
            atomicAdd(offset_bottom_diff + z0 * height * width + y1 * width + x1, g4);
            atomicAdd(offset_bottom_diff + z1 * height * width + y0 * width + x0, g5);
            atomicAdd(offset_bottom_diff + z1 * height * width + y0 * width + x1, g6);
            atomicAdd(offset_bottom_diff + z1 * height * width + y1 * width + x0, g7);
            atomicAdd(offset_bottom_diff + z1 * height * width + y1 * width + x1, g8);
          }  // if
        }    // ix
      }      // iy
    }        // iz
  }          // CUDA_1D_KERNEL_LOOP
}  // RoIAlignBackward

#endif  // ROI_ALIGN_CUDA_KERNEL_CUH
