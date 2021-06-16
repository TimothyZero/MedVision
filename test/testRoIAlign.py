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
import time
import torch

from torchvision.ops import RoIAlign as RoIAlignTorchvision

from medvision.ops import RoIAlign as RoIAlignCuda
from medvision.ops.torch import RoIAlign as RoIAlignTorch

from medvision.io import ImageIO

from medvision.visulaize import volume2tiled


def call_roi_align(name, obj, fs, rois, device='cpu'):
    fs, rois = fs.float().to(device), rois.to(device)

    tic = time.time()
    r = obj((64, 64), 1.0, 1)
    a = r(fs, rois)  # [num_boxes,  C, H, W]
    print(a.shape)
    # print(a.cpu().numpy())
    print(time.time() - tic)

    ImageIO.saveArray(name + '.jpg', a[0].cpu().numpy())


def test2d():
    image, _, _, _ = ImageIO.loadArray("../samples/det_image.jpg")
    image = image[0, ...]
    print(image.shape)

    f = torch.from_numpy(image).int()
    f = f.unsqueeze(0).unsqueeze(0)
    fs1 = torch.cat([1 * f], dim=1)
    fs2 = torch.cat([3 * f], dim=1)
    fs = torch.cat([fs1, fs2], dim=0)

    # annotation is "bbox": [382, 28, 80, 80] => [382, 28, 462-1, 108-1]
    rois = torch.tensor([
                            [0, 379.5, 25.5, 382.5, 28.5],
                        ] * 1000)
    # print(rois.shape)

    call_roi_align('RoIAlignTorchvision', RoIAlignTorchvision, fs, rois)
    call_roi_align('RoIAlignCuda', RoIAlignCuda, fs, rois, device='cuda')
    call_roi_align('RoIAlignTorch', RoIAlignTorch, fs, rois, device='cuda')


def test3d():
    import SimpleITK as sitk
    image = sitk.GetArrayFromImage(sitk.ReadImage("../samples/lung.nii.gz"))  # 30 512 512

    f = torch.from_numpy(image).float()
    f = f.unsqueeze(0).unsqueeze(0)
    fs1 = torch.cat([1 * f], dim=1)
    fs2 = torch.cat([3 * f], dim=1)
    fs = torch.cat([fs1, fs2], dim=0)
    print(fs.shape)

    rois = torch.tensor([[0, 0.0, 0.0, 0.0, 255, 255, 14]])  # xyz order
    print(rois.shape)

    r = RoIAlignCuda((15, 256, 256), 1.0, 1)
    a = r(fs.float().cuda(), rois.cuda())  # [num_boxes,  C, D, H, W]

    volume2tiled(a[0, 0].cpu().numpy(), 'test.roi.jpg', 1)
    volume2tiled(image[0:15, 0:256, 0:256], 'test.crop.jpg', 1)

    print(a.shape)
    # print(a)


if __name__ == "__main__":
    """
    In this test, we can find that RpIAlign will contains the start and stop elements, 
    the roi shape of a bbox [x1,y1,x2,y2] will be [x2-x1+1, y2-y1+1],
    So, while loading a bbox, we should use [start + shape -1] 
    Pay attention to that!
    """

    test2d()
    test3d()
