# -*- coding:utf-8 -*-
from typing import Tuple, Union
import random
import numpy as np
import torch
from torch.nn import functional as F
import scipy.ndimage as ndi

from .base import AugBase
from .cuda_fun_tools import random_noise_2d, random_noise_3d


class Normalize(AugBase):
    """
    Normalize the image to [-1.0, 1.0].

    support segmentation, detection and classification tasks
    support 2D and 3D images
    support forward and backward

    Args:
        mean (sequence): Mean values of each channels.
        std (sequence): Std values of each channels.
    """

    def __init__(self, mean, std, clip=True):
        super().__init__()
        self.always = True
        self.mean = mean
        self.std = std
        self.clip = clip

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(mean={}, std={}, clip={})'.format(self.mean, self.std, self.clip)
        return repr_str

    @property
    def canBackward(self):
        return True

    def _forward_params(self, result):
        self._init_params(result)
        # 3 channel [(128, 128, 128), (128, 128, 128)]
        self.params = [tuple(self.mean), tuple(self.std)]
        result[self.key_name] = self.params

    def _backward_params(self, result):
        params = super()._backward_params(result)
        if params is not None:
            # [(-1, -1, -1), (1/128, 1/128, 1/128)]
            r_mean = - torch.HalfTensor(params[0]) / torch.HalfTensor(params[1])
            r_std = 1 / torch.HalfTensor(params[1])
            self.params = [tuple(r_mean), tuple(r_std)]

    def apply_to_img(self, result):
        image = result['img']
        mean, std = self.params
        assert self.channels == len(mean) == len(std), f"channels = {self.channels}"
        expand = (slice(None),) + (None,) * self.dim
        mean = torch.HalfTensor(mean).to(image.device)[expand]
        std = torch.HalfTensor(std).to(image.device)[expand]
        result['img'] = (image - mean) / std
        if self.clip and self.isForwarding:
            result['img'] = torch.clip(result['img'], -1.0, 1.0)


class MultiNormalize(AugBase):
    """
    Normalize the image to [-1.0, 1.0].

    support segmentation, detection and classification tasks
    support 2D and 3D images
    support forward and backward

    Args:
        means (sequence): Mean values of each channels.
        stds (sequence): Std values of each channels.
    """

    def __init__(self, means, stds, clip=True):
        super().__init__()
        self.always = True
        self.means = means
        self.stds = stds
        self.clip = clip
        assert len(means[0]) == 1, 'only support one channel image'

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(means={}, stds={}, clip={})'.format(self.means, self.stds, self.clip)
        return repr_str

    @property
    def canBackward(self):
        return True

    def _forward_params(self, result):
        self._init_params(result)
        # [[(128, ), (192, )], [(128, ), (192, )]]
        self.params = [self.means, self.stds]
        result[self.key_name] = self.params

    def _backward_params(self, result):
        self._init_params(result)
        params = result.pop(self.key_name, None)
        if params is not None:
            # [[-128/128, -192/192], [1/128, 1/192]]
            self.params = [[], []]
            for mean, std in zip(params[0], params[1]):
                r_mean = - np.array(mean) / np.array(std)
                r_std = 1 / np.array(std)
                self.params[0].append(r_mean[0])
                self.params[1].append(r_std[0])

    def apply_to_img(self, result):
        if self.isForwarding:
            image = result['img']
            means, stds = self.params
            assert self.channels == len(means[0]) == len(stds[0]), f"channels = {self.channels}, it should be same"

            expand = (slice(None),) + (None,) * self.dim
            normalized_images = []
            for mean, std in zip(means, stds):
                mean = torch.HalfTensor(mean).to(image.device)[expand]
                std = torch.HalfTensor(std).to(image.device)[expand]
                normalized_images.append((image - mean) / std)
            img = torch.cat(normalized_images, dim=0)
            if self.clip and self.isForwarding:
                img = torch.clip(img, -1.0, 1.0)
            result['img'] = img
            result['img_shape'] = img.shape
        # else:
        #     img = result['img'].astype(np.float32)
        #     mean, std = self.params
        #     assert self.channels == len(mean) == len(std), f"channels={self.channels}, mean={mean}"
        #     assert img.shape == result['img_shape']
        #     expand = (slice(None),) + (None,) * self.dim
        #     img = (img - np.array(mean, dtype=np.float32)[expand]) / np.array(std, dtype=np.float32)[expand]
        #     img = np.mean(img, axis=0, keepdims=True)
        #     result['img'] = img
        #     result['img_shape'] = img.shape


class RandomBlur(AugBase):
    """
    support segmentation, detection and classification tasks
    support 2D and 3D images
    """

    def __init__(self, p, sigma: float):
        super().__init__()
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
        self.params = self.get_range([0, self.sigma])
        result[self.key_name] = self.params

    def _backward_params(self, result):
        self._init_params(result)
        self.params = True

    def apply_to_img(self, result):
        if self.isForwarding:
            image = result['img']

            sigma = float(self.params)
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
                new_image = F.conv2d(image.unsqueeze(0),
                                     weight=torch.FloatTensor(kernel).type(self.img_type).to(image.device),
                                     padding=half_win).squeeze(0)
            elif self.dim == 3:
                new_image = F.conv3d(image.unsqueeze(0),
                                     weight=torch.FloatTensor(kernel).type(self.img_type).to(image.device),
                                     padding=half_win).squeeze(0)
            else:
                raise NotImplementedError
            result['img'] = new_image


class RandomNoise(AugBase):
    def __init__(self,
                 p: float,
                 method: str = 'uniform',
                 mean: float = 0,
                 std: float = 0.1):
        super().__init__()
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
            result['img'] = random_noise_2d(result['img'].unsqueeze(0),
                                            method=inner_method,
                                            mean=self.mean,
                                            std=self.std,
                                            inplace=True).squeeze(0)
        elif self.dim == 3:
            result['img'] = random_noise_3d(result['img'].unsqueeze(0),
                                            method=inner_method,
                                            mean=self.mean,
                                            std=self.std,
                                            inplace=True).squeeze(0)
        else:
            raise NotImplementedError
