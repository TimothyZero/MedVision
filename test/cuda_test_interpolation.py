# -*- coding:utf-8 -*-
import torch
import matplotlib.pyplot as plt
from scipy import ndimage

from medvision.aug_cuda.cuda_fun_tools import affine_2d, affine_3d


def test2d():
    offset = torch.tensor([
        [1, 4, -3, 3, 1],
        [-3, 2, 5, 4, -3],
        [3, -2, 4, -5, -8],
        [-4, 8, -4, 4, -2],
        [0, 4, 7, 0, 4]
    ]).float().to('cuda')

    dim = 2
    index = torch.FloatTensor([0])
    center = torch.FloatTensor([i / 2 for i in list(offset.shape[::-1])])
    shape = torch.FloatTensor(list(offset.shape[::-1]))
    if dim == 2:
        angles = torch.FloatTensor([0])
    else:
        angles = torch.FloatTensor([0, 0, 0])

    rois = torch.cat([index, center, shape, angles]).unsqueeze(0).to('cuda')
    out_size = tuple([int(i * p) for i, p in zip(offset.shape, [12, 12])])
    spatial_scale = 1
    aligned = True

    image_order0 = affine_2d(
        offset.unsqueeze(0).unsqueeze(0),
        rois,
        out_size,
        spatial_scale,
        1,
        aligned,
        0
    ).squeeze(0)

    image_order1 = affine_2d(
        offset.unsqueeze(0).unsqueeze(0),
        rois,
        out_size,
        spatial_scale,
        1,
        aligned,
        1
    ).squeeze(0)

    image_order3 = affine_2d(
        offset.unsqueeze(0).unsqueeze(0),
        rois,
        out_size,
        spatial_scale,
        1,
        aligned,
        3
    ).squeeze(0)

    zoomed = ndimage.zoom(offset.cpu().numpy(), [12, 12], order=3)

    plt.figure(figsize=(15, 15))
    plt.subplot(221)
    plt.imshow(offset.cpu().numpy())
    plt.subplot(222)
    plt.imshow(image_order0[0].cpu().numpy())
    plt.subplot(223)
    plt.imshow(image_order1[0].cpu().numpy())
    plt.subplot(224)
    plt.imshow(image_order3[0].cpu().numpy())
    plt.show()

    plt.figure(figsize=(20, 10))
    plt.subplot(121)
    plt.imshow(zoomed)
    plt.subplot(122)
    plt.imshow(image_order3[0].cpu().numpy())
    plt.show()


def test3d():
    offset = torch.rand([9, 9, 9]).float().to('cuda') - 0.5

    dim = 3
    index = torch.FloatTensor([0])
    center = torch.FloatTensor([i / 2 for i in list(offset.shape[::-1])])
    shape = torch.FloatTensor(list(offset.shape[::-1]))
    if dim == 2:
        angles = torch.FloatTensor([0])
    else:
        angles = torch.FloatTensor([0, 0, 0])

    rois = torch.cat([index, center, shape, angles]).unsqueeze(0).to('cuda')
    out_size = tuple([int(i * p) for i, p in zip(offset.shape, [12, 12, 12])])
    spatial_scale = 1
    aligned = True

    image_order0 = affine_3d(
        offset.unsqueeze(0).unsqueeze(0),
        rois,
        out_size,
        spatial_scale,
        1,
        aligned,
        0
    ).squeeze(0)

    image_order1 = affine_3d(
        offset.unsqueeze(0).unsqueeze(0),
        rois,
        out_size,
        spatial_scale,
        1,
        aligned,
        1
    ).squeeze(0)

    image_order3 = affine_3d(
        offset.unsqueeze(0).unsqueeze(0),
        rois,
        out_size,
        spatial_scale,
        1,
        aligned,
        3
    ).squeeze(0)

    zoomed = ndimage.zoom(offset.cpu().numpy(), [12, 12, 12], order=3)

    plt.figure(figsize=(15, 15))
    plt.subplot(221)
    plt.imshow(offset[24 // 12].cpu().numpy())
    plt.subplot(222)
    plt.imshow(image_order0[0, 24].cpu().numpy())
    plt.subplot(223)
    plt.imshow(image_order1[0, 24].cpu().numpy())
    plt.subplot(224)
    plt.imshow(image_order3[0, 24].cpu().numpy())
    plt.show()

    plt.figure(figsize=(20, 10))
    plt.subplot(121)
    plt.imshow(zoomed[24])
    plt.subplot(122)
    plt.imshow(image_order3[0, 24].cpu().numpy())
    plt.show()


if __name__ == "__main__":
    test2d()
    test3d()
