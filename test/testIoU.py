# -*- coding:utf-8 -*-
import torch

from medvision.ops import bbox_overlaps_nd

bboxes1 = torch.FloatTensor([
    [0, 0, 10, 10],
    [10, 10, 20, 20],
    [32, 32, 38, 42],
])
bboxes2 = torch.FloatTensor([
    [0, 0, 10, 20],
    [0, 10, 10, 19],
    [10, 10, 20, 20],
])
print(bbox_overlaps_nd(bboxes1.cuda(), bboxes2.cuda()))

bboxes1 = torch.FloatTensor([
    [10., 10., 10., 25., 35., 35.],
    [14., 12., 12., 28., 39., 39.],
    [12., 16., 16., 24., 30., 30.],
    [32., 32., 32., 46., 55., 55.],
    [34., 35., 35., 48., 58., 58.],
    [30., 39., 39., 44., 62., 62.],
    [36., 43., 43., 52., 60., 60.],
])
bboxes2 = torch.FloatTensor([
    [10., 10., 10., 25., 35., 35.],
    [14., 12., 12., 28., 39., 39.],
    [12., 16., 16., 24., 30., 30.],
    [32., 32., 32., 46., 55., 55.],
    [34., 35., 35., 48., 58., 58.],
    [30., 39., 39., 44., 62., 62.],
    [36., 43., 43., 52., 60., 60.],
])
print(bbox_overlaps_nd(bboxes1.cuda(), bboxes2.cuda()))