# -*- coding:utf-8 -*-
import time

from medvision.aug_cuda import CudaRandomBlur, CudaDisplay
from medvision.visulaize import volume2tiled

from load_utils import *


def test2d():
    result, img_filename, seg_filename = load_2d_image_with_seg_with_det()

    transform = CudaRandomBlur(p=1, sigma=[1.0, 3.0])
    tic = time.time()
    transformed_result = transform(result)
    toc = time.time()
    print(transform)
    print(toc - tic)
    print(transform.params)
    print(transformed_result.keys())

    io.imsave(f'{save_to}/{img_filename}.png',
              np.clip(255.0 * transformed_result['img'][0].cpu().numpy(), 0, 255).astype(np.uint8))
    io.imsave(f'{save_to}/{seg_filename}.png',
              np.clip(255.0 * transformed_result['gt_seg'][0].cpu().numpy(), 0, 255).astype(np.uint8))
    print('\n')
    CudaDisplay()(transformed_result)


def test3d():
    result, img_filename, seg_filename = load_3d_image_with_seg()

    transform = CudaRandomBlur(p=1, sigma=[1.0, 3.0])
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
    CudaDisplay()(transformed_result)


if __name__ == "__main__":
    __dir__ = os.path.dirname(os.path.realpath(__file__))
    save_to = os.path.join(__dir__, 'CudaRandomBlur')
    os.makedirs(save_to, exist_ok=True)
    os.chdir(__dir__)

    test2d()
    test3d()
