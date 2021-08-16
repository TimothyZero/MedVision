#pragma once
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

at::Tensor nms_2d_cuda(const at::Tensor boxes, float nms_overlap_thresh);

at::Tensor nms_2d(
    const at::Tensor& dets,
    const at::Tensor& scores,
    const float threshold) {
  if (dets.device().is_cuda()) {

    if (dets.numel() == 0) {
      at::cuda::CUDAGuard device_guard(dets.device());
      return at::empty({0}, dets.options().dtype(at::kLong));
    }
    auto b = at::cat({dets, scores.unsqueeze(1)}, 1);
    return nms_2d_cuda(b, threshold);

  }
  AT_ERROR("Not compiled with CPU support");
//  at::Tensor result = nms_cpu(dets, scores, threshold);
//  return result;
}