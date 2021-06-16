#include "nms_2d_interface.h"
#include "nms_3d_interface.h"
#include "RoIAlign_2d_interface.h"
#include "RoIAlign_3d_interface.h"
#include "deform_pool_2d_interface.h"  // mmdet
#include "deform_conv_2d_interface.h"  // mmdet
#include "deform_conv_3d_interface.h"

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

    m.def("deform_conv_forward_cuda", &deform_conv_forward_cuda,
        "deform forward (CUDA)");
    m.def("deform_conv_backward_input_cuda", &deform_conv_backward_input_cuda,
        "deform_conv_backward_input (CUDA)");
    m.def("deform_conv_backward_parameters_cuda", &deform_conv_backward_parameters_cuda,
        "deform_conv_backward_parameters (CUDA)");

    m.def("modulated_deform_conv_cuda_forward", &modulated_deform_conv_cuda_forward,
        "modulated deform conv forward (CUDA)");
    m.def("modulated_deform_conv_cuda_backward", &modulated_deform_conv_cuda_backward,
        "modulated deform conv backward (CUDA)");

    m.def("deform_conv_forward", &deform_conv_forward, "deform_conv_forward");
    m.def("deform_conv_backward", &deform_conv_backward, "deform_conv_backward");
}