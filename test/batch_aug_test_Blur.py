# -*- coding:utf-8 -*-
import os
import time
import torch
import numpy as np
from skimage import io
from skimage.util import img_as_float32

from medvision.aug_cuda_batch import BatchCudaRandomBlur, Display
from medvision.visulaize import volume2tiled


def test2d():
    img_path = "../samples/21_training.png"
    img_filename = os.path.basename(img_path)
    img = img_as_float32(io.imread(img_path, as_gray=True))

    seg_path = "../samples/21_manual1.png"
    seg_filename = os.path.basename(seg_path)
    seg = img_as_float32(io.imread(seg_path, as_gray=True))

    print('type', img.dtype, 'max', img.max())
    print('type', seg.dtype, 'max', seg.max())

    img = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).cuda()  # batch, channel, *shape
    seg = torch.from_numpy(seg).float().unsqueeze(0).unsqueeze(0).cuda()

    result = {
        'img_dim':    2,
        'img':        torch.cat([img, ] * 4, dim=0),
        'gt_seg':     torch.cat([seg, ] * 4, dim=0),
        'seg_fields': ['gt_seg'],
    }
    order = 3
    transform = BatchCudaRandomBlur(p=1.0, sigma=1.5)
    tic = time.time()
    transformed_result = transform(result)
    toc = time.time()
    print(transform)
    print(toc - tic)
    print(transform.params)
    print(transformed_result.keys())

    for i in range(len(result['img'])):
        io.imsave(f'{save_to}/{img_filename}.b{i}.{order}.png',
                  np.clip(255.0 * transformed_result['img'][i, 0].cpu().numpy(), 0, 255).astype(np.uint8))
        io.imsave(f'{save_to}/{seg_filename}.b{i}.{order}.png',
                  np.clip(255.0 * transformed_result['gt_seg'][i, 0].cpu().numpy(), 0, 255).astype(np.uint8))
    print('\n')
    Display()(transformed_result)


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

    img = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).cuda()
    seg = torch.from_numpy(seg).float().unsqueeze(0).unsqueeze(0).cuda()

    result = {
        'img_dim':    3,
        'img_spacing': (1.0, 1.0, 1.0),
        'img':        torch.cat([img, ] * 2, dim=0),
        'gt_seg':     torch.cat([seg, ] * 2, dim=0),
        'seg_fields': ['gt_seg'],
    }

    order = 3
    transform = BatchCudaRandomBlur(p=1.0, sigma=[0.5, 2.5])
    tic = time.time()
    transformed_result = transform(result)
    toc = time.time()
    print(transform)
    print(toc - tic)
    print(transform.params)
    print(transformed_result.keys())

    for i in range(len(result['img'])):
        volume2tiled(transformed_result['img'][i, 0].cpu().numpy(), f'{save_to}/{img_filename}.b{i}.{order}.png', 10)
        volume2tiled(transformed_result['gt_seg'][i, 0].cpu().numpy(), f'{save_to}/{seg_filename}.b{i}.{order}.png', 10)
    print('\n')
    Display()(transformed_result)

    from medvision.io import ImageIO
    for i in range(len(result['img'])):
        ImageIO.saveArray(f'{save_to}/{img_filename}.b{i}.{order}.nii.gz', transformed_result['img'][i].cpu().numpy())
        ImageIO.saveArray(f'{save_to}/{seg_filename}.b{i}.{order}.gt_seg.nii.gz', transformed_result['gt_seg'][i].cpu().numpy())


if __name__ == "__main__":
    __dir__ = os.path.dirname(os.path.realpath(__file__))
    save_to = os.path.join(__dir__, 'BatchCudaRandomBlur')
    os.makedirs(save_to, exist_ok=True)
    os.chdir(__dir__)

    test2d()
    test3d()
