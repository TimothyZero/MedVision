#ifndef ROI_ALIGN_CUDA_KERNEL_CUH
#define ROI_ALIGN_CUDA_KERNEL_CUH
#include <stdio.h>
#include <float.h>

#include "cuda_helpers.h"
#include "interpolation_helpers_2d.h"

/*** Forward ***/
template <typename scalar_t>
__global__ void roi_align_2d_forward_cuda_kernel(
    const int nthreads, const scalar_t *bottom_data,
    const scalar_t *bottom_rois, const scalar_t spatial_scale,
    const int sampling_ratio, const int order,
    const int channels,
    const int height, const int width,
    const int pooled_height, const int pooled_width,
    scalar_t *top_data) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const scalar_t *offset_bottom_rois = bottom_rois + n * 5;
    int roi_batch_ind = offset_bottom_rois[0];

    // Do not using rounding; this implementation detail is critical
    scalar_t roi_start_w = offset_bottom_rois[1] * spatial_scale;
    scalar_t roi_start_h = offset_bottom_rois[2] * spatial_scale;
    scalar_t roi_end_w = offset_bottom_rois[3] * spatial_scale;
    scalar_t roi_end_h = offset_bottom_rois[4] * spatial_scale;
    scalar_t roi_width = max(roi_end_w - roi_start_w, (scalar_t)1.);
    scalar_t roi_height = max(roi_end_h - roi_start_h, (scalar_t)1.);

    roi_width = max(roi_width, (scalar_t)1.);
    roi_height = max(roi_height, (scalar_t)1.);

    scalar_t bin_size_h = static_cast<scalar_t>(roi_height) / static_cast<scalar_t>(pooled_height);
    scalar_t bin_size_w = static_cast<scalar_t>(roi_width) / static_cast<scalar_t>(pooled_width);

    const scalar_t *offset_bottom_data = bottom_data + (roi_batch_ind * channels + c) * height * width;

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_height / pooled_height);  // e.g., = 2
    int roi_bin_grid_w = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

    // We do average (integral) pooling inside a bin
    const scalar_t count = max(roi_bin_grid_h * roi_bin_grid_w, 1);  // e.g. = 4

    scalar_t output_val = 0.;
    for (int iy = 0; iy < roi_bin_grid_h; iy++) // e.g., iy = 0, 1
    {
      const scalar_t y = roi_start_h + ph * bin_size_h + static_cast<scalar_t>(iy + .5f) * bin_size_h / static_cast<scalar_t>(roi_bin_grid_h); // e.g., 0.5, 1.5
      for (int ix = 0; ix < roi_bin_grid_w; ix++)
      {
        const scalar_t x = roi_start_w + pw * bin_size_w + static_cast<scalar_t>(ix + .5f) * bin_size_w / static_cast<scalar_t>(roi_bin_grid_w);

        scalar_t val;
        if (order == 0) {
          val = nearest_2d_interpolate<scalar_t>(offset_bottom_data, height, width, y, x, index);
        }
        else if (order == 1) {
          val = linear_2d_interpolate<scalar_t>(offset_bottom_data, height, width, y, x, index);
        }
        output_val += val;
      }
    }
    output_val /= count;

    top_data[index] = output_val;
  }
}

/*** Backward ***/
template <typename scalar_t>
__global__ void roi_align_2d_backward_cuda_kernel(
    const int nthreads, const scalar_t *top_diff, const scalar_t *bottom_rois,
    const scalar_t spatial_scale, const int sampling_ratio, const int order,
    const int channels,
    const int height, const int width,
    const int pooled_height, const int pooled_width,
    scalar_t *bottom_diff) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const scalar_t *offset_bottom_rois = bottom_rois + n * 5;
    int roi_batch_ind = offset_bottom_rois[0];

    // Do not round
    scalar_t roi_start_w = offset_bottom_rois[1] * spatial_scale;
    scalar_t roi_start_h = offset_bottom_rois[2] * spatial_scale;
    scalar_t roi_end_w = offset_bottom_rois[3] * spatial_scale;
    scalar_t roi_end_h = offset_bottom_rois[4] * spatial_scale;
    scalar_t roi_width = max(roi_end_w - roi_start_w, (scalar_t)1.);
    scalar_t roi_height = max(roi_end_h - roi_start_h, (scalar_t)1.);

    roi_width = max(roi_width, (scalar_t)1.);
    roi_height = max(roi_height, (scalar_t)1.);

    scalar_t bin_size_h = static_cast<scalar_t>(roi_height) / static_cast<scalar_t>(pooled_height);
    scalar_t bin_size_w = static_cast<scalar_t>(roi_width) / static_cast<scalar_t>(pooled_width);

    scalar_t *offset_bottom_diff =
        bottom_diff + (roi_batch_ind * channels + c) * height * width;

    int top_offset = (n * channels + c) * pooled_height * pooled_width;
    const scalar_t *offset_top_diff = top_diff + top_offset;
    const scalar_t top_diff_this_bin = offset_top_diff[ph * pooled_width + pw];

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_height / pooled_height);  // e.g., = 2
    int roi_bin_grid_w = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

    // We do average (integral) pooling inside a bin
    const scalar_t count = roi_bin_grid_h * roi_bin_grid_w;  // e.g. = 4

    for (int iy = 0; iy < roi_bin_grid_h; iy++) // e.g., iy = 0, 1
    {
      const scalar_t y = roi_start_h + ph * bin_size_h +
          static_cast<scalar_t>(iy + .5f) * bin_size_h / static_cast<scalar_t>(roi_bin_grid_h); // e.g., 0.5, 1.5

      for (int ix = 0; ix < roi_bin_grid_w; ix++)
      {
        const scalar_t x = roi_start_w + pw * bin_size_w +
            static_cast<scalar_t>(ix + .5f) * bin_size_w / static_cast<scalar_t>(roi_bin_grid_w);

        scalar_t w1, w2, w3, w4;
        int x0, x1, y0, y1;
        if (order == 0) {
          nearest_2d_interpolate_gradient<scalar_t>(height, width,
                                                    y, x,
                                                    w1, w2, w3, w4,
                                                    x0, x1,
                                                    y0, y1,
                                                    index);
        }
        else if (order == 1) {
          linear_2d_interpolate_gradient<scalar_t>(height, width,
                                                   y, x,
                                                   w1, w2, w3, w4,
                                                   x0, x1,
                                                   y0, y1,
                                                   index);
        }

        scalar_t g1 = top_diff_this_bin * w1 / count;
        scalar_t g2 = top_diff_this_bin * w2 / count;
        scalar_t g3 = top_diff_this_bin * w3 / count;
        scalar_t g4 = top_diff_this_bin * w4 / count;

        if (x0 >= 0 && x1 >= 0 && y0 >= 0 && y1 >= 0) {
          atomicAdd(offset_bottom_diff + y0 * width + x0, g1);
          atomicAdd(offset_bottom_diff + y0 * width + x1, g2);
          atomicAdd(offset_bottom_diff + y1 * width + x0, g3);
          atomicAdd(offset_bottom_diff + y1 * width + x1, g4);
        }  // if
      }    // ix
    }      // iy
  }        // CUDA_1D_KERNEL_LOOP
}  // RoIAlignBackward

#endif  // ROI_ALIGN_CUDA_KERNEL_CUH
