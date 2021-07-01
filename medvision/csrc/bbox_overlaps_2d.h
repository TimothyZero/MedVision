#pragma once
#include <torch/extension.h>

void bbox_overlaps_2d_cuda(const at::Tensor bboxes1, const at::Tensor bboxes2,
                                    at::Tensor ious, const int mode,
                                    const bool aligned, const int offset);

void bbox_overlaps_2d(const at::Tensor bboxes1, const at::Tensor bboxes2, at::Tensor ious,
                   const int mode, const bool aligned, const int offset) {
  if (bboxes1.device().is_cuda()) {
#ifdef WITH_CUDA
//    CHECK_CUDA_INPUT(bboxes1);
//    CHECK_CUDA_INPUT(bboxes2);
//    CHECK_CUDA_INPUT(ious);

    bbox_overlaps_2d_cuda(bboxes1, bboxes2, ious, mode, aligned, offset);
#else
    AT_ERROR("bbox_overlaps is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("bbox_overlaps is not implemented on CPU");
  }
}
