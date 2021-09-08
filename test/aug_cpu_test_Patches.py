# -*- coding:utf-8 -*-
import time

from medvision.aug_cpu import Patches, Display
from medvision.visulaize import volume2tiled

from load_utils import *


def test2d():
    result, img_filename, seg_filename = load_2d_image_with_seg(to_cuda=False)

    transform = Patches(patch_size=(300, 300))
    tic = time.time()
    transformed_result = transform(result)
    toc = time.time()
    print(transform)
    print(toc - tic)
    print(transform.params)
    print(transformed_result.keys())

    volume2tiled(transformed_result['patches_img'][:, 0], f'{save_to}/{img_filename}.png', 1)
    volume2tiled(transformed_result['patches_gt_seg'][:, 0], f'{save_to}/{seg_filename}.png', 1)

    print('\n')
    Display()(transformed_result)


def test3d():
    result, img_filename, seg_filename = load_3d_image_with_seg(to_cuda=False)

    transform = Patches(patch_size=(256, 256, 256))
    tic = time.time()
    transformed_result = transform(result)
    toc = time.time()
    print(transform)
    print(toc - tic)
    print(transform.params)
    print(transformed_result.keys())

    volume2tiled(transformed_result['patches_img'][0, 0], f'{save_to}/{img_filename}.png', 10)
    volume2tiled(transformed_result['patches_gt_seg'][0, 0], f'{save_to}/{seg_filename}.png', 10)
    print('\n')
    Display()(transformed_result)


if __name__ == "__main__":
    __dir__ = os.path.dirname(os.path.realpath(__file__))
    save_to = os.path.join(__dir__, 'CpuPatches')
    os.makedirs(save_to, exist_ok=True)
    os.chdir(__dir__)

    test2d()
    test3d()
