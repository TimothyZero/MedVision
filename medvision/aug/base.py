# -*- coding:utf-8 -*-
import re
import os
import subprocess
import time
from datetime import datetime
import gc
from abc import abstractmethod
from typing import Union, Iterable, Tuple, List
import numpy as np
import random
from copy import deepcopy

import torch.cuda


def get_gpu():
    try:
        with open(os.devnull, 'w') as devnull:
            gpus = subprocess.check_output([f'nvidia-smi'],
                                           stderr=devnull).decode().rstrip('\r\n').split('\n')
        gpu = re.search('\d+MiB', [i for i in gpus if str(os.getpid()) in i][0])[0]
        return gpu
    except Exception as e:
        return f'N/A({str(os.getpid())}-Err:{e})'


class AugBase(object):
    def __init__(self):
        self.p = 0.
        self.latitude = 1.0
        self.always = False
        self.dim = None
        self.params = None
        self.isForwarding = None  # if forward or backward
        self.name = self.__class__.__name__
        self.key_name = self.name + '_params'  # dict key name saving in result

    @property
    def canBackward(self):
        return False

    @property
    def repeats(self):
        return 1

    @staticmethod
    def get_range(val_range, bias=0):
        if isinstance(val_range, (int, float)):
            val_range = [- val_range, val_range]
        assert isinstance(val_range, (list, tuple)) and len(val_range) == 2
        return bias + np.random.uniform(*val_range)
        # return bias + val_range

    @staticmethod
    def to_tuple(value: Union[int, float, Iterable[int], Iterable[float]],
                 length: int = 1
                 ) -> Union[Tuple[int], Tuple[float]]:
        """
        If value is int or float, return (val,) * length
        to_tuple(1, length=3) -> (1, 1, 1)

        If value is an iterable, length is ignored and tuple(value) is returned
        to_tuple([1, 2], length=3) -> (1, 2)
        """
        try:
            iter(value)
            value = tuple(value)
        except TypeError:
            value = length * (value,)
        assert len(value) == length, f'Dimension not match. Required is {length}D but value is {len(value)}D'
        return value

    def _init_params(self, result):
        """ initial some params """
        self.dim = self.dim if self.dim else result['img_dim']
        self.img_type = result['img'].dtype
        if 'img' in result.keys():
            self.array_shape = tuple(result['img'].shape)
        else:
            self.array_shape = tuple(result['img_shape'])
        self.image_axes = tuple(range(1, self.dim + 1))
        self.channels = self.array_shape[0]
        self.image_shape = self.array_shape[1:]
        self.params = None

    @abstractmethod
    def _forward_params(self, result: dict):
        # raise NotImplementedError
        pass

    @abstractmethod
    def _backward_params(self, result: dict):
        # raise NotImplementedError
        pass

    def setLatitude(self, val):
        self.latitude = val

    def getLatitude(self):
        return self.latitude

    def apply_to_img(self, result: dict):
        pass

    def apply_to_cls(self, result: dict):
        pass

    def apply_to_seg(self, result: dict):
        pass

    def apply_to_det(self, result: dict):
        pass

    def _pre_forward(self, result: dict):
        assert isinstance(result, dict), f'A dict is required but got a {type(result)}!'
        self._debug_ = result.get('_debug_', False)
        self._tic_ = time.time()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        return result

    def _forward(self, result: dict):
        # print(self.__class__.__name__, "forward...")
        self.isForwarding = True
        assert self.always or self.p > 0., f'self.always={self.always} or self.p={self.p} is needed.'
        if self.always or random.random() <= self.p * self.latitude:
            # print("Doing", self.name)
            self.params = None
            self._forward_params(result)
            self.apply_to_img(result)
            self.apply_to_cls(result)
            self.apply_to_seg(result)
            self.apply_to_det(result)
            # print(self.params)
        return result

    def _post_forward(self, result: dict):
        assert isinstance(result, dict), f'A dict is required but got a {type(result)}!'
        if 'history' not in result.keys():
            result['history'] = []
        if 'time' not in result.keys():
            result['time'] = []
        if 'memory' not in result.keys():
            result['memory'] = []
        _start = datetime.fromtimestamp(self._tic_).strftime('%H:%M:%S.%f')
        _memory = torch.cuda.max_memory_allocated() / 1024 /1024 / 8
        result['history'].append(self.name)
        result['time'].append(f"{self.name}-{_start}-{time.time() - self._tic_:.03f}s")
        result['memory'].append(f"{self.name}-{get_gpu()}-{_memory:.02f}MB")
        gc.collect()
        torch.cuda.empty_cache()
        return result

    def forward(self, result: dict):
        return self._post_forward(self._forward(self._pre_forward(result)))

    def multi_forward(self, result_list: Union[List[dict], dict]):
        if self.repeats > 1:
            result = [result_list] if isinstance(result_list, dict) else result_list
            return [self.forward(deepcopy(r)) for r in result for _ in range(self.repeats)]
        else:
            assert isinstance(result_list, list) and self.repeats == 1
            return [self.forward(result) for result in result_list]

    def _pre_backward(self, result: dict):
        time_cost = result['time'].pop()
        last = result['history'].pop()
        assert self.name in time_cost, f'{self.name} {time_cost}'
        assert last == self.name, f'require params to be {self.name}, but the last is {last}'
        return result

    def _backward(self, result: dict):
        assert isinstance(result, dict)
        # print(self.__class__.__name__, "backward...")
        self.isForwarding = False
        if self.canBackward:
            self.params = None
            self._backward_params(result)
            if self.params is not None:  # because of the prob is not 1, sometimes, it will not do transforms on data
                # print("Doing")
                # print(self.params)
                self.apply_to_img(result)
                self.apply_to_cls(result)
                self.apply_to_seg(result)
                self.apply_to_det(result)
        return result

    def _post_backward(self, result: dict):
        gc.collect()
        return result

    def backward(self, result):
        return self._post_backward(self._backward(self._pre_backward(result)))

    def multi_backward(self, result_list: list):
        return [self.backward(result) for result in result_list]

    def __call__(self, result: Union[dict, List[dict]], forward=True):
        """return a same type object as input object"""
        if forward:
            if isinstance(result, dict) and self.repeats == 1:
                return self.forward(result)
            elif isinstance(result, list) or self.repeats > 1:
                return self.multi_forward(result)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
            # # while using backward, it means result has already been a list of dict
            # if isinstance(result, list):
            #     return self.multi_backward(result)
            # else:
            #     return self.backward(result)
