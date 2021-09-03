import os
import time
import json

from medvision.aug_cuda import *


def test2d():
    pre = CudaLoadPrepare()
    d = CudaDisplay()
    v = CudaViewer()

    result = pre(os.path.abspath("../samples/det_image.jpg"),
                 det=json.load(open(os.path.abspath("../samples/det_bboxes.json"))))

    # result = pre(os.path.abspath("../samples/21_training.png"),
    #              os.path.abspath("../samples/21_manual1.png"))

    d(result)

    tic = time.time()
    load_img = CudaLoadImageFromFile(to_float16=True)
    load_ann = CudaLoadAnnotations(with_det=True)
    # load_ann = LoadAnnotations(with_seg=True)
    result = load_img(result)
    result = load_ann(result)
    print('load and cuda:', time.time() - tic)
    d(result)

    transforms = [
        CudaNormalize(mean=(128, 128, 128), std=(128, 128, 128)),
        # Normalize(mean=(128, 128, 128, 128), std=(128, 128, 128, 128)),
        CudaRandomFlip(p=1.0),
        CudaPad(size=(666, 666)),
        CudaCropRandom(patch_size=(512, 512)),
        CudaResize(factor=(2.0, 2.0)),
        # RandomRotate(p=1.0, rotate=20),
        # RandomScale(p=1.0, scale=0.2),
        # RandomShift(p=1.0, shift=0.2),
        CudaRandomAffine(p=1.0, scale=0.1, shift=0.1, rotate=20),
        CudaPad(size=(1660, 1660)),
        CudaRandomElasticDeformationFast(p=1.0),
        CudaRandomBlur(p=1.0, sigma=3.0),
        CudaRandomNoise(p=1.0, method='uniform', mean=0.0, std=0.2),
        CudaCollect(keys=['img', 'gt_det']),
        # Collect(keys=['img', 'gt_seg']),
    ]
    pipeline = CudaForwardCompose(transforms)
    print(pipeline)

    tic = time.time()
    result = pipeline(result)
    print('ForwardCompose:', time.time() - tic)
    d(result)
    v(result)


def test3d():
    pre = CudaLoadPrepare()
    d = CudaDisplay()
    v = CudaViewer()

    result = pre(os.path.abspath("../samples/luna16_iso_crop_img.nii.gz"),
                 seg=os.path.abspath("../samples/luna16_iso_crop_lung.nii.gz"))

    d(result)

    tic = time.time()
    load_img = CudaLoadImageFromFile(to_float16=True)
    load_ann = CudaLoadAnnotations(with_seg=True)
    result = load_img(result)
    result = load_ann(result)
    print('load and cuda:', time.time() - tic)
    d(result)
    # v(result)

    transforms = [
        CudaNormalize(mean=(-400,), std=(700,), clip=True),
        CudaRandomFlip(p=1.0),
        CudaPad(size=(256, 256, 256)),
        CudaCropRandom(patch_size=(256, 256, 256)),
        CudaResize(factor=(1.2, 1.2, 1.2)),
        CudaRandomAffine(p=1.0, scale=0.1, shift=0.1, rotate=20),
        CudaRandomBlur(p=1.0, sigma=1.0),
        CudaRandomElasticDeformationFast(p=1.0),
        # RandomNoise(p=1.0, method='uniform', mean=0.0, std=0.1),
        CudaCollect(keys=['img', 'gt_seg']),
    ]
    pipeline = CudaForwardCompose(transforms)
    print(pipeline)

    tic = time.time()
    result = pipeline(result)
    print('ForwardCompose:', time.time() - tic)
    d(result)
    v(result)


if __name__ == "__main__":
    __dir__ = os.path.dirname(os.path.realpath(__file__))
    os.chdir(__dir__)

    test2d()
    test3d()
