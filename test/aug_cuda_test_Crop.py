# -*- coding:utf-8 -*-
import time

from medvision.aug_cuda import CudaCropRandom, CudaCropCenter, CudaCropForeground, \
    CudaCropDet, CudaCropFirstDet, CudaCropFirstDetOnly, \
    CudaDisplay, CudaViewer
from medvision.visulaize import volume2tiled

from load_utils import *


def test2d():
    result, img_filename, seg_filename = load_2d_image_with_seg_with_det()

    transform = USED_CROP(patch_size=(384, 440), times=4)
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
        io.imsave(f'{save_to}/{USED_CROP.__name__}.{i}.{img_filename}.png',
                  np.minimum(255, 255.0 * t['img'][0].cpu().numpy()).astype(np.uint8))
        io.imsave(f'{save_to}/{USED_CROP.__name__}.{i}.{seg_filename}.png',
                  np.minimum(255, 255.0 * t['gt_seg'][0].cpu().numpy()).astype(np.uint8))
    print('\n')
    CudaDisplay()(transformed_result)
    CudaViewer(save_dir=save_to)(transformed_result)


def test3d():
    result, img_filename, seg_filename = load_3d_image_with_seg()

    transform = USED_CROP(patch_size=(50, 184, 184))
    tic = time.time()
    transformed_result = transform(result)
    toc = time.time()
    print(transform)
    print(toc - tic)
    print(transform.params)
    print(transformed_result.keys())

    volume2tiled(transformed_result['img'][0].cpu().numpy(),
                 f'{save_to}/{USED_CROP.__name__}.{img_filename}.png', 10)
    volume2tiled(transformed_result['gt_seg'][0].cpu().numpy(),
                 f'{save_to}/{USED_CROP.__name__}.{seg_filename}.png', 10)
    print('\n')
    CudaDisplay()(transformed_result)


if __name__ == "__main__":
    __dir__ = os.path.dirname(os.path.realpath(__file__))
    save_to = os.path.join(__dir__, 'CudaCrop')
    os.makedirs(save_to, exist_ok=True)
    os.chdir(__dir__)

    USED_CROP = CudaCropForeground

    test2d()
    if "Det" not in USED_CROP.__name__:
        test3d()
