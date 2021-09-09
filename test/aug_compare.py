# -*- coding:utf-8 -*-
from medvision.aug_cpu import *
from medvision.aug_cuda import *
from medvision.aug_cuda_batch import *
from medvision.visulaize import volume2tiled

from load_utils import *


def test2d_transform(cpu_transform, cuda_transform, cuda_batch_transform):
    cpu_result, img_filename, seg_filename = load_2d_image_with_seg(to_cuda=False)
    cuda_result, img_filename, seg_filename = load_2d_image_with_seg(to_cuda=True)
    cuda_batch_result, img_filename, seg_filename = load_2d_image_with_seg(to_cuda=True, batch=4)

    cpu_transformed_result = cpu_transform(cpu_result)
    cuda_transformed_result = cuda_transform(cuda_result)
    cuda_batch_transformed_result = cuda_batch_transform(cuda_batch_result)

    io.imsave(f'{save_to}/{cpu_transform.__class__.__name__}{img_filename}.cpu.png',
              np.clip(255.0 * cpu_transformed_result['img'][0], 0, 255).astype(np.uint8))
    io.imsave(f'{save_to}/{cpu_transform.__class__.__name__}{seg_filename}.cpu.png',
              np.clip(255.0 * cpu_transformed_result['gt_seg'][0], 0, 255).astype(np.uint8))

    io.imsave(f'{save_to}/{cpu_transform.__class__.__name__}{img_filename}.cuda.png',
              np.clip(255.0 * cuda_transformed_result['img'][0].cpu().numpy(), 0, 255).astype(np.uint8))
    io.imsave(f'{save_to}/{cpu_transform.__class__.__name__}{seg_filename}.cuda.png',
              np.clip(255.0 * cuda_transformed_result['gt_seg'][0].cpu().numpy(), 0, 255).astype(np.uint8))

    io.imsave(f'{save_to}/{cpu_transform.__class__.__name__}{img_filename}.cuda_batch.png',
              np.clip(255.0 * cuda_batch_transformed_result['img'][0, 0].cpu().numpy(), 0, 255).astype(np.uint8))
    io.imsave(f'{save_to}/{cpu_transform.__class__.__name__}{seg_filename}.cuda_batch.png',
              np.clip(255.0 * cuda_batch_transformed_result['gt_seg'][0, 0].cpu().numpy(), 0, 255).astype(np.uint8))


def test2d():
    transforms = [
        Resize(factor=(0.8, 0.8)),
        CudaResize(factor=(0.8, 0.8)),
        BatchCudaResize(factor=(0.8, 0.8)),
    ]

    # transforms = [
    #     RandomNoise(p=1.0),
    #     CudaRandomNoise(p=1.0),
    #     BatchCudaRandomNoise(p=1.0),
    # ]

    test2d_transform(*transforms)


def test3d_transform(cpu_transform, cuda_transform, cuda_batch_transform):
    cpu_result, img_filename, seg_filename = load_3d_image_with_seg(to_cuda=False)
    cuda_result, img_filename, seg_filename = load_3d_image_with_seg(to_cuda=True)
    cuda_batch_result, img_filename, seg_filename = load_3d_image_with_seg(to_cuda=True, batch=4)

    cpu_transformed_result = cpu_transform(cpu_result)
    cuda_transformed_result = cuda_transform(cuda_result)
    cuda_batch_transformed_result = cuda_batch_transform(cuda_batch_result)

    volume2tiled(cpu_transformed_result['img'][0],
                 f'{save_to}/{cpu_transform.__class__.__name__}{img_filename}.cpu.png', 10)
    volume2tiled(cpu_transformed_result['gt_seg'][0],
                 f'{save_to}/{cpu_transform.__class__.__name__}{seg_filename}.cpu.png', 10)

    volume2tiled(cuda_transformed_result['img'][0].cpu().numpy(),
                 f'{save_to}/{cpu_transform.__class__.__name__}{img_filename}.cuda.png', 10)
    volume2tiled(cuda_transformed_result['gt_seg'][0].cpu().numpy(),
                 f'{save_to}/{cpu_transform.__class__.__name__}{seg_filename}.cuda.png', 10)

    volume2tiled(cuda_batch_transformed_result['img'][0, 0].cpu().numpy(),
                 f'{save_to}/{cpu_transform.__class__.__name__}{img_filename}.cuda_batch.png', 10)
    volume2tiled(cuda_batch_transformed_result['gt_seg'][0, 0].cpu().numpy(),
                 f'{save_to}/{cpu_transform.__class__.__name__}{seg_filename}.cuda_batch.png', 10)


def test3d():
    # transforms = [
    #     Resize(factor=(0.8, 0.8, 0.8)),
    #     CudaResize(factor=(0.8, 0.8, 0.8)),
    #     BatchCudaResize(factor=(0.8, 0.8, 0.8)),
    # ]

    method = 'normal'
    transforms = [
        RandomNoise(p=1.0, method=method, std=100),
        CudaRandomNoise(p=1.0, method=method, std=100),
        BatchCudaRandomNoise(p=1.0, method=method, std=100),
    ]

    test3d_transform(*transforms)


if __name__ == "__main__":
    __dir__ = os.path.dirname(os.path.realpath(__file__))
    save_to = os.path.join(__dir__, 'Compare')
    os.makedirs(save_to, exist_ok=True)
    os.chdir(__dir__)

    test2d()
    test3d()