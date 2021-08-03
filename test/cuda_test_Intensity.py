# -*- coding:utf-8 -*-
import os
import time
import torch
import numpy as np
from skimage import io
from skimage.util import img_as_float32
from matplotlib.colors import Normalize

from medvision.aug.cuda_fun_tools import random_noise_2d, random_noise_3d
from medvision.visulaize import volume2tiled


def test2d():
    img_path = "../samples/21_training.png"
    img_filename = os.path.basename(img_path)
    img = img_as_float32(io.imread(img_path, as_gray=True))
    img = (img - 0.5) / 0.5
    print('type', img.dtype, 'min', img.min(), 'max', img.max())

    img = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).cuda()

    method = 0
    mean = 0.2
    std = 0.5
    tic = time.time()
    out = random_noise_2d(img, method, mean, std)
    toc = time.time()
    print(img.max())
    print(out.max(), out.min())
    print(toc - tic, '\n')

    io.imsave(f'{save_to}/{img_filename}.intensity.{method}.png',
              (255 * Normalize(-1, 1, clip=True)(out[0, 0].cpu().numpy())).astype(np.uint8))

    method = 1
    mean = 0.5
    std = 0.2
    tic = time.time()
    out = random_noise_2d(img, method, mean, std)
    toc = time.time()
    print(img.max())
    print(out.max(), out.min())
    print(toc - tic, '\n')

    io.imsave(f'{save_to}/{img_filename}.intensity.{method}.png',
              (255 * Normalize(-1, 1, clip=True)(out[0, 0].cpu().numpy())).astype(np.uint8))


def test3d():
    import SimpleITK as sitk

    img_path = "../samples/luna16_iso_crop_img.nii.gz"
    img_filename = os.path.basename(img_path)
    img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
    img = (img + 400) / 700
    print('type', img.dtype, 'min', img.min(), 'max', img.max())
    print(img.shape)

    img = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).cuda()

    method = 0
    mean = 0
    std = 0.2
    tic = time.time()
    out = random_noise_3d(img, method, mean, std)
    toc = time.time()
    print(img.max())
    print(out.max(), out.min())
    print(toc - tic, '\n')

    volume2tiled(out[0, 0].cpu().numpy(), f'{save_to}/{img_filename}.intensity.{method}.png')

    method = 1
    mean = 0
    std = 0.2
    tic = time.time()
    out = random_noise_3d(img.clone(), method, mean, std)
    toc = time.time()
    print(img.max())
    print(out.max(), out.min())
    print(toc - tic, '\n')

    volume2tiled(out[0, 0].cpu().numpy(), f'{save_to}/{img_filename}.intensity.{method}.png')


if __name__ == "__main__":
    __dir__ = os.path.dirname(os.path.realpath(__file__))
    save_to = os.path.join(__dir__, 'Intensity')
    os.makedirs(save_to, exist_ok=True)
    os.chdir(__dir__)

    test2d()
    test3d()