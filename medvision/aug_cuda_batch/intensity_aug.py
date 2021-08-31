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
from ..aug_cuda.cuda_fun_tools import random_noise_2d, random_noise_3d


class BatchCudaRandomNoise(BatchCudaAugBase):
    def __init__(self, p,
                 method: str = 'uniform',
                 mean: float = 0,
                 std: float = 0.1):
        super(BatchCudaRandomNoise, self).__init__()
        self.supported = ['normal', 'uniform']
        assert method in self.supported, f"method should be one of {self.supported}"
        self.p = p
        self.method = method
        self.mean = mean
        self.std = std

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(p={}, method={}, mean={}, std={})'.format(self.p, self.method, self.mean, self.std)
        return repr_str

    @property
    def canBackward(self):
        return True

    def _forward_params(self, result):
        self._init_params(result)
        self.params = (self.mean, self.std)
        result[self.key_name] = self.params

    def _backward_params(self, result):
        pass

    def apply_to_img(self, result):
        inner_method = self.supported.index(self.method)
        if self.dim == 2:
            result['img'] = random_noise_2d(result['img'],
                                            method=inner_method,
                                            mean=self.mean,
                                            std=self.std,
                                            inplace=True)
        elif self.dim == 3:
            result['img'] = random_noise_3d(result['img'],
                                            method=inner_method,
                                            mean=self.mean,
                                            std=self.std,
                                            inplace=True)
        else:
            raise NotImplementedError


class BatchCudaRandomBlur(BatchCudaAugBase):
    def __init__(self, p, sigma: Union[float, List]):
        super(BatchCudaRandomBlur, self).__init__()
        self.p = p
        self.sigma = sigma
        self.truncate = 3.0

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(p={}, sigma={})'.format(self.p, self.sigma)
        return repr_str

    @property
    def canBackward(self):
        return True

    def _forward_params(self, result):
        self._init_params(result)
        self.params = self.get_range(self.sigma, always_pos=True)  # [ for _ in range(self.batch)]
        # self.params = [self.get_range(self.sigma, always_pos=True) for _ in range(self.batch)]
        result[self.key_name] = self.params

    def _backward_params(self, result):
        self._init_params(result)
        self.params = True

    def apply_to_img(self, result):
        if self.isForwarding:
            image = result['img']

            sigma = max(float(self.params), 0)
            half_win = int(self.truncate * sigma + 0.5)
            kernel_size = 2 * half_win + 1

            # fill pixel or voxel at center [half_win, half_win] or [half_win, half_win, half_win]
            kernel = np.zeros([kernel_size] * self.dim)
            kernel[tuple([half_win] * self.dim)] = 1

            kernel = ndi.gaussian_filter(kernel, sigma, mode='constant')
            kernel = np.stack([kernel] * self.channels)
            kernel = np.stack([kernel] * self.channels)
            # filter one by one
            for i in range(self.channels):
                for j in range(self.channels):
                    if i != j:
                        kernel[i, j] = 0
            if self.dim == 2:
                new_image = F.conv2d(image,
                                     weight=torch.FloatTensor(kernel).type(self.img_type).to(image.device),
                                     padding=half_win)
            elif self.dim == 3:
                new_image = F.conv3d(image,
                                     weight=torch.FloatTensor(kernel).type(self.img_type).to(image.device),
                                     padding=half_win)
            else:
                raise NotImplementedError
            result['img'] = new_image
