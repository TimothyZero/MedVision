# -*- coding:utf-8 -*-
import time

from medvision.aug_cuda import CudaCropRandomWithAffine, CudaDisplay, CudaViewer
from medvision.visulaize import volume2tiled


from load_utils import *


def test2d():
    result, img_filename, seg_filename = load_2d_image_with_seg_with_det()

    transform = CudaCropRandomWithAffine(patch_size=(380, 420), scale=0.2, shift=0.2, rotate=[20, 40], times=4)
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
    CudaDisplay()(transformed_result)
    CudaViewer(save_dir=save_to)(transformed_result)


def test3d():
    result, img_filename, seg_filename = load_3d_image_with_seg_with_det()

    transform = CudaCropRandomWithAffine(patch_size=(25, 400, 400), scale=0.2, shift=0.25, rotate=[20, 40])
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
    CudaViewer(save_dir=save_to)(transformed_result)


if __name__ == "__main__":
    __dir__ = os.path.dirname(os.path.realpath(__file__))
    save_to = os.path.join(__dir__, 'CudaCropRandomWithAffine')
    os.makedirs(save_to, exist_ok=True)
    os.chdir(__dir__)
    
    test2d()
    test3d()
