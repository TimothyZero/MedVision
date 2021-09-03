# -*- coding:utf-8 -*-
import time

from medvision.aug_cuda_batch import BatchCudaRandomNoise, Display
from medvision.visulaize import volume2tiled

from load_utils import *


def test2d():
    result, img_filename, seg_filename = load_2d_image_with_seg(batch=4)

    transform = BatchCudaRandomNoise(p=1.0)
    tic = time.time()
    transformed_result = transform(result)
    toc = time.time()
    print(transform)
    print(toc - tic)
    print(transform.params)
    print(transformed_result.keys())

    for i in range(len(result['img'])):
        io.imsave(f'{save_to}/{img_filename}.b{i}.png',
                  np.clip(255.0 * transformed_result['img'][i, 0].cpu().numpy(), 0, 255).astype(np.uint8))
        io.imsave(f'{save_to}/{seg_filename}.b{i}.png',
                  np.clip(255.0 * transformed_result['gt_seg'][i, 0].cpu().numpy(), 0, 255).astype(np.uint8))
    print('\n')
    Display()(transformed_result)


def test3d():
    result, img_filename, seg_filename = load_3d_image_with_seg(batch=4)

    transform = BatchCudaRandomNoise(p=1.0)
    tic = time.time()
    transformed_result = transform(result)
    toc = time.time()
    print(transform)
    print(toc - tic)
    print(transform.params)
    print(transformed_result.keys())

    for i in range(len(result['img'])):
        volume2tiled(transformed_result['img'][i, 0].cpu().numpy(), f'{save_to}/{img_filename}.b{i}.png', 10)
        volume2tiled(transformed_result['gt_seg'][i, 0].cpu().numpy(), f'{save_to}/{seg_filename}.b{i}.png', 10)
    print('\n')
    Display()(transformed_result)

    from medvision.io import ImageIO
    for i in range(len(result['img'])):
        ImageIO.saveArray(f'{save_to}/{img_filename}.b{i}.nii.gz', transformed_result['img'][i].cpu().numpy())
        ImageIO.saveArray(f'{save_to}/{seg_filename}.b{i}.gt_seg.nii.gz', transformed_result['gt_seg'][i].cpu().numpy())


if __name__ == "__main__":
    __dir__ = os.path.dirname(os.path.realpath(__file__))
    save_to = os.path.join(__dir__, 'BatchCudaRandomNoise')
    os.makedirs(save_to, exist_ok=True)
    os.chdir(__dir__)

    test2d()
    test3d()
