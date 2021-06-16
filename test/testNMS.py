#  Copyright (c) 2020. The Medical Image Computing (MIC) Lab, 陶豪毅
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import numpy as np
import torch
import time

from medvision.ops import nms_nd


def test2D():
    iou_thr_per_dim = 0.7
    dets_2d = np.array([[10., 10., 25., 35., 0.99],
                        [14., 12., 28., 39., 0.95],
                        [12., 16., 24., 30., 0.52],
                        [32., 32., 46., 55., 0.55],
                        [34., 35., 48., 58., 0.50],
                        [30., 39., 44., 62., 0.45],
                        [36., 43., 52., 60., 0.30]], dtype=np.float32)
    dets_2d = torch.from_numpy(dets_2d).cuda()

    """ CUDA """
    start = time.time()
    keep, suppressed = nms_nd(dets_2d, iou_thr_per_dim ** 2)
    print(time.time() - start)
    print(keep)
    print(suppressed)
    print(suppressed[keep])


def test3D():
    iou_thr_per_dim = 0.7
    dets_3d = np.array([[10., 10., 10., 25., 35., 35., 0.99],
                        [14., 12., 12., 28., 39., 39., 0.95],
                        [12., 16., 16., 24., 30., 30., 0.52],
                        [32., 32., 32., 46., 55., 55., 0.55],
                        [34., 35., 35., 48., 58., 58., 0.50],
                        [30., 39., 39., 44., 62., 62., 0.45],
                        [36., 43., 43., 52., 60., 60., 0.30]], dtype=np.float32)
    dets_3d = torch.from_numpy(dets_3d).cuda()

    """ CUDA """
    start = time.time()
    keep, suppressed = nms_nd(dets_3d, iou_thr_per_dim ** 3)
    print(time.time() - start)
    print(keep)
    print(suppressed)
    print(suppressed[keep])


if __name__ == "__main__":
    """
    For 2d and 3d nmsNd_cuda, the iou threshold should be different!
    iou_thr_per_dim may be useful.
    
    For softnms, linear decay or gaussian decay
    """
    test2D()
    test3D()
