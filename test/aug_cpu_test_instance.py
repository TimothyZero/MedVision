# -*- coding:utf-8 -*-
import time

from medvision.aug_cpu import Instance2BBoxConversion, Display, Viewer
from medvision.visulaize import volume2tiled

from load_utils import *


def test2d():
    result, img_filename, seg_filename = load_2d_image_with_seg_with_det(to_cuda=False)

    transform = Instance2BBoxConversion()
    tic = time.time()
    transformed_result = transform(result)
    toc = time.time()
    print(transform)
    print(toc - tic)
    print(transform.params)
    print(transformed_result.keys())

    print('\n')
    Display()(transformed_result)
    Viewer(save_dir=save_to)(transformed_result)


def test3d():
    result, img_filename, seg_filename = load_3d_image_with_seg_with_det(to_cuda=False)

    transform = Instance2BBoxConversion()
    tic = time.time()
    transformed_result = transform(result)
    toc = time.time()
    print(transform)
    print(toc - tic)
    print(transform.params)
    print(transformed_result.keys())

    print('\n')
    Display()(transformed_result)
    Viewer(save_dir=save_to)(transformed_result)


if __name__ == "__main__":
    __dir__ = os.path.dirname(os.path.realpath(__file__))
    save_to = os.path.join(__dir__, 'CpuInstance')
    os.makedirs(save_to, exist_ok=True)
    os.chdir(__dir__)

    test2d()
    test3d()
