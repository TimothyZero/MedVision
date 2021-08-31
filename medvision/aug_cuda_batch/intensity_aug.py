import math
import time
import warnings
import itertools
from typing import Union, Iterable, Tuple, List
import torch
import torch.nn.functional as F
import numpy as np
import random
import SimpleITK as sitk
import scipy.ndimage as ndi

from .base import BatchCudaAugBase


class BatchCudaRandomFlip(BatchCudaAugBase):
    def __init__(self, p,
                 axes: List = [0]):
        super(BatchCudaRandomFlip, self).__init__()
        self.p = p
        self.axes = axes

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(p={}, axes={})'.format(self.p, self.axes)
        return repr_str


# class BatchCudaRandomNoise(BatchCudaAugBase)
# class BatchCudaRandomBlur(BatchCudaAugBase)
