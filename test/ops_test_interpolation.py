# -*- coding:utf-8 -*-
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import ndimage

from medvision.ops.cuda_fun_tools import affine_2d, affine_3d


def test2d():
    offset = torch.rand([7, 7]).float().to('cuda') - 0.5

    zoomed0 = ndimage.zoom(offset.cpu().numpy(), [10, 10], order=0)
    zoomed1 = ndimage.zoom(offset.cpu().numpy(), [10, 10], order=1)
    zoomed3 = ndimage.zoom(offset.cpu().numpy(), [10, 10], order=3)

    dim = 2
    index = torch.FloatTensor([0])
    center = torch.FloatTensor([i / 2 for i in list(offset.shape[::-1])])
    shape = torch.FloatTensor(list(offset.shape[::-1])) - 1
    if dim == 2:
        angles = torch.FloatTensor([0])
    else:
        angles = torch.FloatTensor([0, 0, 0])

    rois = torch.cat([index, center, shape, angles]).unsqueeze(0).to('cuda')
    out_size = tuple([int(i * p) for i, p in zip(offset.shape, [10, 10])])
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
    ).squeeze(0).cpu().numpy()

    image_order1 = affine_2d(
        offset.unsqueeze(0).unsqueeze(0),
        rois,
        out_size,
        spatial_scale,
        1,
        aligned,
        1
    ).squeeze(0).cpu().numpy()

    image_order3 = affine_2d(
        offset.unsqueeze(0).unsqueeze(0),
        rois,
        out_size,
        spatial_scale,
        1,
        aligned,
        3
    ).squeeze(0).cpu().numpy()

    plt.figure(figsize=(15, 15))
    plt.subplot(221)
    plt.imshow(offset.cpu().numpy())
    plt.subplot(222)
    plt.imshow(zoomed0)
    plt.subplot(223)
    plt.imshow(zoomed1)
    plt.subplot(224)
    plt.imshow(zoomed3)
    plt.show()

    plt.figure(figsize=(15, 15))
    plt.subplot(221)
    plt.imshow(offset.cpu().numpy())
    plt.subplot(222)
    plt.imshow(image_order0[0])
    plt.subplot(223)
    plt.imshow(image_order1[0])
    plt.subplot(224)
    plt.imshow(image_order3[0])
    plt.show()

    all_img = np.vstack([np.hstack([zoomed0, zoomed1, zoomed3]),
                         np.hstack([image_order0[0], image_order1[0], image_order3[0]])])
    plt.figure(figsize=(30, 20))
    plt.imshow(all_img)
    plt.xticks(range(0, all_img.shape[1], 2))
    plt.yticks(range(0, all_img.shape[0], 2))
    plt.grid()
    plt.tight_layout()
    plt.show()


def test3d():
    offset = torch.rand([9, 9, 9]).float().to('cuda') - 0.5

    dim = 3
    index = torch.FloatTensor([0])
    center = torch.FloatTensor([i / 2 for i in list(offset.shape[::-1])])
    shape = torch.FloatTensor(list(offset.shape[::-1])) - 1
    if dim == 2:
        angles = torch.FloatTensor([0])
    else:
        angles = torch.FloatTensor([0, 0, 0])

    rois = torch.cat([index, center, shape, angles]).unsqueeze(0).to('cuda')
    out_size = tuple([int(i * p) for i, p in zip(offset.shape, [10, 10, 10])])
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

    zoomed = ndimage.zoom(offset.cpu().numpy(), [10, 10, 10], order=3)

    plt.figure(figsize=(15, 15))
    plt.subplot(221)
    plt.imshow(offset[30 // 10].cpu().numpy())
    plt.subplot(222)
    plt.imshow(image_order0[0, 30].cpu().numpy())
    plt.subplot(223)
    plt.imshow(image_order1[0, 30].cpu().numpy())
    plt.subplot(224)
    plt.imshow(image_order3[0, 30].cpu().numpy())
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
