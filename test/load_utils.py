# -*- coding:utf-8 -*-
import os
import torch
import numpy as np
from skimage import io
from skimage.util import img_as_float32


def load_2d_image_with_seg_with_det(norm=False, downsample=1, to_cuda=True, batch=None):
    img_path = "../samples/det_image.jpg"
    img_filename = os.path.basename(img_path)
    img = img_as_float32(io.imread(img_path, as_gray=True))[None, ...]
    if norm:
        img = (img - 0.5) / 0.5

    seg_path = "../samples/det_seg.png"
    seg_filename = os.path.basename(seg_path)
    seg = img_as_float32(io.imread(seg_path, as_gray=True))[None, ...]

    img = img[:, ::downsample, ::downsample]
    seg = seg[:, ::downsample, ::downsample]

    det = np.array([
        [382, 28, 462, 108, 1, 1],
        [32, 248, 117, 303, 2, 1]
    ])

    print('type', img.dtype, 'max', img.max())
    if to_cuda:
        img = torch.from_numpy(img).float().cuda()
        seg = torch.from_numpy(seg).float().cuda()
        det = torch.from_numpy(det).float().cuda()

    if batch is None:
        result = {
            'img_dim':    2,
            'filename':   img_filename,
            'img':        img,
            'gt_det':     det,
            'det_fields': ['gt_det'],
            'gt_seg':     seg,
            'seg_fields': ['gt_seg']
        }
    else:
        assert to_cuda
        assert isinstance(batch, int)
        result = {
            'img_dim':    2,
            'filename':   img_filename,
            'img':        torch.cat([img.unsqueeze(0), ] * batch, dim=0),
            'gt_det':     torch.cat([det.unsqueeze(0), ] * batch, dim=0),
            'det_fields': ['gt_det'],
            'gt_seg':     torch.cat([seg.unsqueeze(0), ] * batch, dim=0),
            'seg_fields': ['gt_seg']
        }
    return result, img_filename, seg_filename


def load_2d_image_with_seg(norm=False, downsample=1, to_cuda=True, batch=None):
    img_path = "../samples/21_training.png"
    img_filename = os.path.basename(img_path)
    img = img_as_float32(io.imread(img_path, as_gray=True))[None, ...]
    if norm:
        img = (img - 0.5) / 0.5

    seg_path = "../samples/21_manual1.png"
    seg_filename = os.path.basename(seg_path)
    seg = img_as_float32(io.imread(seg_path, as_gray=True))[None, ...]

    img = img[:, ::downsample, ::downsample]
    seg = seg[:, ::downsample, ::downsample]

    print('type', img.dtype, 'max', img.max())
    print('type', seg.dtype, 'max', seg.max())

    if to_cuda:
        img = torch.from_numpy(img).float().cuda()
        seg = torch.from_numpy(seg).float().cuda()

    if batch is None:
        result = {
            'img_dim':    2,
            'filename':   img_filename,
            'img':        img,
            'gt_seg':     seg,
            'seg_fields': ['gt_seg'],
        }
    else:
        assert to_cuda
        assert isinstance(batch, int)
        result = {
            'img_dim':    2,
            'filename':   img_filename,
            'img':        torch.cat([img.unsqueeze(0), ] * batch, dim=0),
            'gt_seg':     torch.cat([seg.unsqueeze(0), ] * batch, dim=0),
            'seg_fields': ['gt_seg']
        }
    return result, img_filename, seg_filename


def load_3d_image_with_seg_with_det(norm=False, downsample=1, to_cuda=True, batch=None):
    import SimpleITK as sitk

    img_path = "../samples/lung.nii.gz"
    img_filename = os.path.basename(img_path)
    img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))[None, ...]
    if norm:
        img = (img + 400) / 700

    seg_path = "../samples/nodule_seg.nii.gz"
    seg_filename = os.path.basename(seg_path)
    seg = sitk.GetArrayFromImage(sitk.ReadImage(seg_path)).astype(int)[None, ...]

    det = np.array([
        [185, 185, 19, 205, 205, 29, 1, 1],
    ])

    img = img[:, ::downsample, ::downsample, ::downsample]
    seg = seg[:, ::downsample, ::downsample, ::downsample]

    print('type', img.dtype, 'max', img.max())
    print('type', seg.dtype, 'max', seg.max())

    if to_cuda:
        img = torch.from_numpy(img).float().cuda()
        seg = torch.from_numpy(seg).float().cuda()
        det = torch.from_numpy(det).float().cuda()

    if batch is None:
        result = {
            'img_dim':    3,
            'filename':   img_filename,
            'img':        img,
            'gt_det':     det,
            'det_fields': ['gt_det'],
            'gt_seg':     seg,
            'seg_fields': ['gt_seg'],
        }
    else:
        assert to_cuda
        assert isinstance(batch, int)
        result = {
            'img_dim':    3,
            'filename':   img_filename,
            'img':        torch.cat([img.unsqueeze(0), ] * batch, dim=0),
            'gt_det':     torch.cat([det.unsqueeze(0), ] * batch, dim=0),
            'det_fields': ['gt_det'],
            'gt_seg':     torch.cat([seg.unsqueeze(0), ] * batch, dim=0),
            'seg_fields': ['gt_seg']
        }
    return result, img_filename, seg_filename


def load_3d_image_with_seg(norm=False, downsample=1, to_cuda=True, batch=None):
    import SimpleITK as sitk

    img_path = "../samples/luna16_iso_crop_img.nii.gz"
    img_filename = os.path.basename(img_path)
    img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))[None, ...]
    if norm:
        img = (img + 400) / 700

    seg_path = "../samples/luna16_iso_crop_lung.nii.gz"
    seg_filename = os.path.basename(seg_path)
    seg = sitk.GetArrayFromImage(sitk.ReadImage(seg_path))[None, ...]

    img = img[:, ::downsample, ::downsample, ::downsample]
    seg = seg[:, ::downsample, ::downsample, ::downsample]

    print('type', img.dtype, 'max', img.max())
    print('type', seg.dtype, 'max', seg.max())

    if to_cuda:
        img = torch.from_numpy(img).float().cuda()
        seg = torch.from_numpy(seg).float().cuda()

    if batch is None:
        result = {
            'img_dim':    3,
            'filename':   img_filename,
            'img':        img,
            'gt_seg':     seg,
            'seg_fields': ['gt_seg'],
        }
    else:
        assert to_cuda
        assert isinstance(batch, int)
        result = {
            'img_dim':    3,
            'filename':   img_filename,
            'img':        torch.cat([img.unsqueeze(0), ] * batch, dim=0),
            'gt_seg':     torch.cat([seg.unsqueeze(0), ] * batch, dim=0),
            'seg_fields': ['gt_seg']
        }
    return result, img_filename, seg_filename
