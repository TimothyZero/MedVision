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
import os
import math
import time
import torch
import numpy as np
from skimage import io
from skimage.util import img_as_float32

from torchvision.ops import RoIAlign as RoIAlignTorchvision

# 2d is exactly the same with RoIAlignTorchvision
from medvision.ops import RoIAlign as RoIAlignCuda
from medvision.ops import RoIAlignRotated
from medvision.ops import MyRoIAlign
from medvision.ops.pytorch_ops import RoIAlign as RoIAlignTorchFun
from medvision.visulaize import volume2tiled


def call_roi_align_2d(name, obj, fs, rois, device='cuda', nearest=False):
    try:
        fs, rois = fs.half().to(device), rois.half().to(device)

        tic = time.time()
        r = obj(output_size=(256, 256), spatial_scale=1.0, sampling_ratio=1)
        if nearest:
            a = r(fs, rois, order=0)  # [num_boxes,  C, H, W]
        else:
            a = r(fs, rois)  # [num_boxes,  C, H, W]
        print(name)
        print(time.time() - tic)
        print(a.shape)
        # print(a.cpu().numpy())
        print('mean:', a[0, 0].mean())
        print('')
        io.imsave(f'{save_to}/{name}.png', np.minimum(255, 255.0 * a[0, 0].cpu().numpy()).astype(np.uint8))
    except Exception as e:
        print(e)


def test2d(img_path, nearest=False):
    filename = os.path.basename(img_path)
    image = img_as_float32(io.imread(img_path, as_gray=True))

    print(img_path)
    print('type', image.dtype, 'max', image.max())
    print(image.shape, '\n')

    f = torch.from_numpy(image).float()
    f = f.unsqueeze(0).unsqueeze(0)
    fs1 = torch.cat([1 * f], dim=1)
    fs2 = torch.cat([3 * f], dim=1)
    fs = torch.cat([fs1, fs2], dim=0)

    # roi is : batch_index, x1, y1, x2, y2
    rois = torch.tensor([
                            [0, 161.5, 111.5, 417.5, 367.5],
                        ] * 1000)

    call_roi_align_2d(filename + 'RoIAlignTorchvision', RoIAlignTorchvision, fs, rois, device='cuda')
    call_roi_align_2d(filename + 'RoIAlignTorchFun', RoIAlignTorchFun, fs, rois, nearest=nearest)
    call_roi_align_2d(filename + 'MyRoIAlign', MyRoIAlign, fs, rois, nearest=nearest)
    call_roi_align_2d(filename + 'RoIAlignCuda', RoIAlignCuda, fs, rois)

    # rotated_rois is : batch_index, center_x, center_y, w, h, angle
    rotated_rois1 = torch.tensor([
                                     [0, 290., 240., 256.0, 256.0, 0.0],
                                 ] * 1000)
    rotated_rois2 = torch.tensor([
                                     [0, 290., 240., 256.0, 256.0, math.pi * 90.0/180],
                                 ] * 1000)
    call_roi_align_2d(filename + 'RoIAlignRotated.1', RoIAlignRotated, fs, rotated_rois1, nearest=nearest)
    call_roi_align_2d(filename + 'RoIAlignRotated.2', RoIAlignRotated, fs, rotated_rois2, nearest=nearest)

    # image coord is y, x order
    io.imsave(f'{save_to}/{filename}.crop.png', (255 * image[112:368, 162:418]).astype(np.uint8))


def call_roi_align_3d(name, obj, fs, rois, device='cuda', nearest=False):
    try:
        fs, rois = fs.float().to(device), rois.to(device)

        tic = time.time()
        r = obj(output_size=(30, 356, 356), spatial_scale=1.0, sampling_ratio=1)
        if nearest:
            a = r(fs, rois, order=0)  # [num_boxes,  C, D, H, W]
        else:
            a = r(fs, rois)  # [num_boxes,  C, D, H, W]
        print(name)
        print(time.time() - tic)
        print(a.shape)
        # print(a.max())
        print('')
        volume2tiled(a[0, 0].cpu().numpy(), f'{save_to}/{name}.png', 1)
    except Exception as e:
        print(e)


def test3d(img_path, nearest=False):
    import SimpleITK as sitk

    filename = os.path.basename(img_path)
    image = sitk.GetArrayFromImage(sitk.ReadImage(img_path))

    f = torch.from_numpy(image).float()
    f = f.unsqueeze(0).unsqueeze(0)
    fs1 = torch.cat([1 * f], dim=1)
    fs2 = torch.cat([3 * f], dim=1)
    fs = torch.cat([fs1, fs2], dim=0)

    rois = torch.tensor([[0, 9.5, 9.5, 59.5, 187.5, 187.5, 89.5]])  # xyz order
    call_roi_align_3d(filename + 'RoIAlignCuda', RoIAlignCuda, fs, rois)
    call_roi_align_3d(filename + 'MyRoIAlign', MyRoIAlign, fs, rois, nearest=nearest)

    rotated_rois1 = torch.tensor([[0, 99., 99., 75., 178, 178, 30, 0, 0, 0]])
    rotated_rois2 = torch.tensor([[0, 99., 99., 75., 178, 178, 30, 0, 0, math.pi * -45/180]])
    rotated_rois3 = torch.tensor([[0, 99., 99., 75., 178, 178, 30, math.pi * 45/180, 0, math.pi * -45/180]])
    call_roi_align_3d(filename + 'RoIAlignRotated3d.1', RoIAlignRotated, fs, rotated_rois1, nearest=nearest)
    call_roi_align_3d(filename + 'RoIAlignRotated3d.2', RoIAlignRotated, fs, rotated_rois2, nearest=nearest)
    call_roi_align_3d(filename + 'RoIAlignRotated3d.3', RoIAlignRotated, fs, rotated_rois3, nearest=nearest)

    volume2tiled(image[60:90, 10:188, 10:188], f'{save_to}/{filename}.crop.png', 1)


if __name__ == "__main__":
    """
    In this test, we can find that RpIAlign will contains the start and stop elements, 
    the roi shape of a bbox [x1,y1,x2,y2] will be [x2-x1+1, y2-y1+1],
    So, while loading a bbox, we should use [start + shape -1] 
    Pay attention to that!
    """
    __dir__ = os.path.dirname(os.path.realpath(__file__))
    save_to = os.path.join(__dir__, 'RoIAlign')
    os.makedirs(save_to, exist_ok=True)
    os.chdir(__dir__)

    # # will save the same images after crop or roi align
    test2d("../samples/21_training.png")
    test2d("../samples/21_manual1.png", nearest=True)

    test3d("../samples/luna16_iso_crop_img.nii.gz")
    test3d("../samples/luna16_iso_crop_lung.nii.gz", nearest=True)
