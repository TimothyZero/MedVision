# -*- coding:utf-8 -*-
import json
import os
import time
import torch
import numpy as np
from skimage import io
from skimage.util import img_as_float32

from medvision.aug import CropRandomWithAffine, Display, Viewer
from medvision.visulaize import volume2tiled


def test2d():
    img_path = "../samples/det_image.jpg"
    img_filename = os.path.basename(img_path)
    img = img_as_float32(io.imread(img_path, as_gray=True))

    gt_det = np.array([
        [382, 28, 462, 108, 1, 1],
        [62, 248, 117, 303, 2, 1]
    ])

    print('type', img.dtype, 'max', img.max())

    img = torch.from_numpy(img).float().unsqueeze(0).cuda()

    result = {
        'filename': img_filename,
        'img_dim': 2,
        'img':     img,
        'gt_det':  gt_det,
        'det_fields': ['gt_det'],
        'gt_seg': (img > 0.5).clone(),
        'seg_fields': ['gt_seg']
    }

    transform = CropRandomWithAffine(patch_size=(380, 420), scale=0.2, shift=0.2, rotate=40, times=4)
    tic = time.time()
    transformed_result = transform(result)
    toc = time.time()
    print(transform)
    print(toc - tic)
    print(transform.params)
    if isinstance(transformed_result, dict):
        transformed_result = [transformed_result]
    for i, t in enumerate(transformed_result):
        print(t.keys())
        io.imsave(f'{save_to}/{img_filename}.{i}.png',
                  np.minimum(255, 255.0 * t['img'][0].cpu().numpy()).astype(np.uint8))
    print('\n')
    Display()(transformed_result)
    Viewer(save_dir=save_to)(transformed_result)


def test3d():
    import SimpleITK as sitk

    img_path = "../samples/luna16_iso_crop_img.nii.gz"
    img_filename = os.path.basename(img_path)
    img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))

    seg_path = "../samples/luna16_iso_crop_lung.nii.gz"
    seg_filename = os.path.basename(seg_path)
    seg = sitk.GetArrayFromImage(sitk.ReadImage(seg_path))

    print('type', img.dtype, 'max', img.max())
    print('type', seg.dtype, 'max', seg.max())

    img = torch.from_numpy(img).float().unsqueeze(0).cuda()
    seg = torch.from_numpy(seg).float().unsqueeze(0).cuda()

    result = {
        'img_dim': 3,
        'img':     img,
        'gt_seg':  seg,
        'seg_fields': ['gt_seg'],
    }

    transform = CropRandomWithAffine(patch_size=(128, 200, 200), scale=0.2, shift=0.25, rotate=40)
    tic = time.time()
    transformed_result = transform(result)
    toc = time.time()
    print(transform)
    print(toc - tic)
    print(transform.params)
    print(transformed_result.keys())

    volume2tiled(transformed_result['img'][0].cpu().numpy(), f'{save_to}/{img_filename}.png', 10)
    volume2tiled(transformed_result['gt_seg'][0].cpu().numpy(), f'{save_to}/{seg_filename}.png', 10)
    print('\n')
    Display()(transformed_result)


if __name__ == "__main__":
    __dir__ = os.path.dirname(os.path.realpath(__file__))
    save_to = os.path.join(__dir__, 'CropAffine')
    os.makedirs(save_to, exist_ok=True)
    os.chdir(__dir__)
    
    test2d()
    test3d()
