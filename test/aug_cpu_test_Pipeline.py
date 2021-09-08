# -*- coding:utf-8 -*-
import time

import numpy as np
from medvision.aug_cpu import *
from medvision.visulaize import volume2tiled

from load_utils import *


def test2d():
    result, img_filename, seg_filename = load_2d_image_with_seg(to_cuda=False)

    transform = ForwardCompose([
        Resize(factor=(2.0, 2.0)),
        RandomNoise(p=1.0),
        RandomFlip(p=1.0),
        RandomGamma(p=1.0, gamma=0.2),
        RandomRotate(p=1.0, angle=40)
    ])
    tic = time.time()
    transformed_result = transform(result)
    toc = time.time()
    print(transform)
    print(toc - tic)
    print(transformed_result.keys())

    io.imsave(f'{save_to}/{img_filename}.png',
              np.clip(255.0 * transformed_result['img'][0], 0, 255).astype(np.uint8))
    io.imsave(f'{save_to}/{seg_filename}.png',
              np.clip(255.0 * transformed_result['gt_seg'][0], 0, 255).astype(np.uint8))

    print('\n')
    Display()(transformed_result)


def test3d():
    result, img_filename, seg_filename = load_3d_image_with_seg(to_cuda=False)

    transform = ForwardCompose([
        Resize(factor=(0.8, 0.8, 0.8)),
        RandomNoise(p=1.0),
        RandomFlip(p=1.0),
        RandomGamma(p=1.0, gamma=0.2),
        RandomRotate(p=1.0, angle=40)
    ])
    tic = time.time()
    transformed_result = transform(result)
    toc = time.time()
    print(transform)
    print(toc - tic)
    print(transformed_result.keys())

    volume2tiled(transformed_result['img'][0], f'{save_to}/{img_filename}.png', 10)
    volume2tiled(transformed_result['gt_seg'][0], f'{save_to}/{seg_filename}.png', 10)
    print('\n')
    Display()(transformed_result)


if __name__ == "__main__":
    __dir__ = os.path.dirname(os.path.realpath(__file__))
    save_to = os.path.join(__dir__, 'CpuPipeline')
    os.makedirs(save_to, exist_ok=True)
    os.chdir(__dir__)

    test2d()
    test3d()
