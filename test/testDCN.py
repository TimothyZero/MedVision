import torch
import numpy as np

from medvision.ops import DeformRoIPooling2dPack, ModulatedDeformConv2dPack, DeformConv3dPack, \
    DeformConv2dPack, ModulatedDeformRoIPooling2dPack


def test_2d():
    x_2d = torch.rand((1, 3, 64, 64)).cuda().half()

    dcn = DeformConv2dPack(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1).cuda().half()
    print(dcn(x_2d).shape)
    dcn = ModulatedDeformConv2dPack(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1).cuda().half()
    print(dcn(x_2d).shape)

    # rois = torch.tensor(np.array([[0, 1, 1, 14., 4.], [0, 12, 14, 24., 24.]]), dtype=torch.float).cuda()
    #
    # r = DeformRoIPooling2dPack(spatial_scale=1, out_size=7, out_channels=3, no_trans=True)
    # print(r(x_2d, rois).shape)
    # r = ModulatedDeformRoIPooling2dPack(spatial_scale=1, out_size=7, out_channels=3, no_trans=True)
    # print(r(x_2d, rois).shape)


def test_3d():
    x_3d = torch.rand((1, 3, 64, 64, 64)).half().cuda()

    dcn = DeformConv3dPack(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1).cuda().half()
    print(dcn(x_3d).shape)


if __name__ == '__main__':
    # test_2d()
    test_3d()

