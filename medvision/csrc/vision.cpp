#include "nms_2d.h"
#include "nms_3d.h"
#include "RoIAlign_2d.h"
#include "RoIAlign_3d.h"
#include "deform_conv_2d.h"
#include "deform_conv_3d.h"
#include "modulated_deform_conv_2d.h"
#include "modulated_deform_conv_3d.h"
#include "deform_pool_2d.h"
#include "bbox_overlaps_2d.h"
#include "bbox_overlaps_3d.h"

#ifdef WITH_CUDA
#include <cuda.h>
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("nms_2d", &nms_2d, "non-maximum suppression for 2d");
    m.def("nms_3d", &nms_3d, "non-maximum suppression for 3d");

    m.def("roi_align_2d", &roi_align_2d, "ROIAlign 2D in c++ and/or cuda");
    m.def("roi_align_3d", &roi_align_3d, "ROIAlign 3D in c++ and/or cuda");

    m.def("deform_psroi_pooling_cuda_forward", &deform_psroi_pooling_cuda_forward,
        "deform psroi pooling forward(CUDA)");
    m.def("deform_psroi_pooling_cuda_backward", &deform_psroi_pooling_cuda_backward,
        "deform psroi pooling backward(CUDA)");

    m.def("deform_conv_2d_forward", &deform_conv_2d_forward, "deform_conv_2d_forward");
    m.def("deform_conv_2d_backward", &deform_conv_2d_backward, "deform_conv_2d_backward");

    m.def("deform_conv_3d_forward", &deform_conv_3d_forward, "deform_conv_3d_forward");
    m.def("deform_conv_3d_backward", &deform_conv_3d_backward, "deform_conv_3d_backward");

    m.def("modulated_deform_conv_2d_forward", &modulated_deform_conv_2d_forward, "modulated_deform_conv_2d_forward");
    m.def("modulated_deform_conv_2d_backward", &modulated_deform_conv_2d_backward, "modulated_deform_conv_2d_backward");

    m.def("modulated_deform_conv_3d_forward", &modulated_deform_conv_3d_forward, "modulated_deform_conv_3d_forward");
    m.def("modulated_deform_conv_3d_backward", &modulated_deform_conv_3d_backward, "modulated_deform_conv_3d_backward");

    m.def("bbox_overlaps_2d", &bbox_overlaps_2d, "bbox_overlaps_2d");
    m.def("bbox_overlaps_3d", &bbox_overlaps_3d, "bbox_overlaps_3d");
}