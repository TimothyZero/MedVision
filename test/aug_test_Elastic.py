# -*- coding:utf-8 -*-
import os
import time
import torch
import numpy as np
from skimage import io
from skimage.util import img_as_float32
from matplotlib.colors import Normalize

from medvision.aug import RandomElasticDeformation, RandomElasticDeformationFast, Display
from medvision.visulaize import volume2tiled
from medvision.io import ImageIO


def test2d():
    img_path = "../samples/21_training.png"
    img_filename = os.path.basename(img_path)
    img = img_as_float32(io.imread(img_path, as_gray=True))

    seg_path = "../samples/21_manual1.png"
    seg_filename = os.path.basename(seg_path)
    seg = img_as_float32(io.imread(seg_path, as_gray=True))

    print('type', img.dtype, 'max', img.max())
    print('type', seg.dtype, 'max', seg.max())

    img = torch.from_numpy(img).float().unsqueeze(0).cuda()
    seg = torch.from_numpy(seg).int().unsqueeze(0).cuda()

    io.imsave(f'{save_to}/{img_filename}.raw.png',
              (255 * Normalize(0, 1, clip=True)(img[0].cpu().numpy())).astype(np.uint8))

    result = {
        'img_dim':    2,
        'img':        img,
        'gt_seg':     seg,
        'seg_fields': ['gt_seg'],
        'det_fields': [],
    }

    transform = USED_CLASS(p=1)
    tic = time.time()
    transformed_result = transform(result)
    toc = time.time()
    print(transform)
    print(toc - tic)
    print(transformed_result.keys())

    io.imsave(f'{save_to}/{img_filename}.png',
              (255 * Normalize(0, 1, clip=True)(transformed_result['img'][0].cpu().numpy())).astype(np.uint8))
    io.imsave(f'{save_to}/{seg_filename}.png',
              np.clip(255.0 * transformed_result['gt_seg'][0].cpu().numpy(), 0, 255).astype(np.uint8))

    print('\n')
    Display()(transformed_result)


def test3d():
    import SimpleITK as sitk

    img_path = "../samples/luna16_iso_crop_img.nii.gz"
    img_filename = os.path.basename(img_path)
    img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
    img = (img + 400) / 700

    seg_path = "../samples/luna16_iso_crop_lung.nii.gz"
    seg_filename = os.path.basename(seg_path)
    seg = sitk.GetArrayFromImage(sitk.ReadImage(seg_path))

    print('type', img.dtype, 'max', img.max())
    print('type', seg.dtype, 'max', seg.max())

    img = torch.from_numpy(img).float().unsqueeze(0).cuda()
    seg = torch.from_numpy(seg).float().unsqueeze(0).cuda()

    volume2tiled(img[0].cpu().numpy(), f'{save_to}/{img_filename}.raw.png', 10)

    result = {
        'img_dim':    3,
        'img':        img,
        'gt_seg':     seg,
        'seg_fields': ['gt_seg'],
    }

    transform = USED_CLASS(p=1)
    tic = time.time()
    transformed_result = transform(result)
    toc = time.time()
    print(transform)
    print(toc - tic)
    # print(transform.params)
    print(transformed_result.keys())

    volume2tiled(transformed_result['img'][0].cpu().numpy(), f'{save_to}/{img_filename}.png', 10)
    volume2tiled(transformed_result['gt_seg'][0].cpu().numpy(), f'{save_to}/{seg_filename}.png', 10)
    print('\n')
    Display()(transformed_result)

    ImageIO.saveArray(f'{save_to}/{img_filename}', transformed_result['img'].cpu().numpy())
    ImageIO.saveArray(f'{save_to}/{seg_filename}', transformed_result['gt_seg'].cpu().numpy())


if __name__ == "__main__":
    __dir__ = os.path.dirname(os.path.realpath(__file__))
    save_to = os.path.join(__dir__, 'RandomElasticDeformation')
    os.makedirs(save_to, exist_ok=True)
    os.chdir(__dir__)

    USED_CLASS = RandomElasticDeformationFast

    test2d()
    test3d()