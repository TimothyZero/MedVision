import torch
import numpy as np
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler

from medvision.ops import DeformRoIPooling2dPack, ModulatedDeformRoIPooling2dPack, \
    DeformConv2dPack, ModulatedDeformConv2dPack, \
    DeformConv3dPack, ModulatedDeformConv3dPack


def test_2d():
    x_2d = torch.rand((1, 3, 16, 16), requires_grad=True).cuda()
    seg = torch.ones((1, 16, 16, 16)).cuda()
    loss = torch.nn.BCEWithLogitsLoss()

    # cn = DeformConv2dPack
    cn = ModulatedDeformConv2dPack
    # cn = torch.nn.Conv2d
    dcn = cn(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1).cuda()
    print(dcn)

    scaler = GradScaler()
    op = Adam(lr=0.001, params=dcn.parameters())

    for i in range(100):

        with autocast():
            out = dcn(x_2d)
            l = loss(out, seg)

        print(out.shape, out.type(), l, l.type())

        op.zero_grad()
        scaler.scale(l).backward()
        scaler.step(op)
        scaler.update()


def test_3d():

    x_3d = torch.rand((1, 3, 5, 5, 5), requires_grad=True).cuda()
    seg = torch.ones((1, 7, 5, 5, 5)).cuda()
    loss = torch.nn.BCEWithLogitsLoss()

    # cn = DeformConv3dPack
    cn = ModulatedDeformConv3dPack
    # cn = nn.Conv3d
    dcn = cn(in_channels=3, out_channels=7, kernel_size=3, stride=1, padding=1).cuda()
    print(dcn)

    scaler = GradScaler()
    op = Adam(lr=0.001, params=dcn.parameters())

    for i in range(100):

        with autocast():
            out = dcn(x_3d)
            l = loss(out, seg)

        print(out.shape, out.type(), l, l.type())

        op.zero_grad()
        scaler.scale(l).backward()
        scaler.step(op)
        scaler.update()


if __name__ == '__main__':
    test_2d()
    test_3d()

