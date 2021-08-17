import os
import time
import json

from medvision.aug_cuda import *


def test2d():
    pre = LoadPrepare()
    d = Display()
    v = Viewer()

    result = pre(os.path.abspath("../samples/det_image.jpg"),
                 det=json.load(open(os.path.abspath("../samples/det_bboxes.json"))))

    # result = pre(os.path.abspath("../samples/21_training.png"),
    #              os.path.abspath("../samples/21_manual1.png"))

    d(result)

    tic = time.time()
    load_img = LoadImageFromFile(to_float16=True)
    load_ann = LoadAnnotations(with_det=True)
    # load_ann = LoadAnnotations(with_seg=True)
    result = load_img(result)
    result = load_ann(result)
    print('load and cuda:', time.time() - tic)
    d(result)

    transforms = [
        Normalize(mean=(128, 128, 128), std=(128, 128, 128)),
        # Normalize(mean=(128, 128, 128, 128), std=(128, 128, 128, 128)),
        RandomFlip(p=1.0),
        Pad(size=(666, 666)),
        CropRandom(patch_size=(512, 512)),
        Resize(factor=(2.0, 2.0)),
        # RandomRotate(p=1.0, rotate=20),
        # RandomScale(p=1.0, scale=0.2),
        # RandomShift(p=1.0, shift=0.2),
        RandomAffine(p=1.0, scale=0.1, shift=0.1, rotate=20),
        Pad(size=(1660, 1660)),
        RandomElasticDeformationFast(p=1.0),
        RandomBlur(p=1.0, sigma=3.0),
        RandomNoise(p=1.0, method='uniform', mean=0.0, std=0.2),
        Collect(keys=['img', 'gt_det']),
        # Collect(keys=['img', 'gt_seg']),
    ]
    pipeline = ForwardCompose(transforms)
    print(pipeline)

    tic = time.time()
    result = pipeline(result)
    print('ForwardCompose:', time.time() - tic)
    d(result)
    v(result)


def test3d():
    pre = LoadPrepare()
    d = Display()
    v = Viewer()

    result = pre(os.path.abspath("../samples/luna16_iso_crop_img.nii.gz"),
                 seg=os.path.abspath("../samples/luna16_iso_crop_lung.nii.gz"))

    d(result)

    tic = time.time()
    load_img = LoadImageFromFile(to_float16=True)
    load_ann = LoadAnnotations(with_seg=True)
    result = load_img(result)
    result = load_ann(result)
    print('load and cuda:', time.time() - tic)
    d(result)
    # v(result)

    transforms = [
        Normalize(mean=(-400,), std=(700,), clip=True),
        RandomFlip(p=1.0),
        Pad(size=(256, 256, 256)),
        CropRandom(patch_size=(256, 256, 256)),
        Resize(factor=(1.2, 1.2, 1.2)),
        RandomAffine(p=1.0, scale=0.1, shift=0.1, rotate=20),
        RandomBlur(p=1.0, sigma=1.0),
        RandomElasticDeformationFast(p=1.0),
        # RandomNoise(p=1.0, method='uniform', mean=0.0, std=0.1),
        Collect(keys=['img', 'gt_seg']),
    ]
    pipeline = ForwardCompose(transforms)
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
