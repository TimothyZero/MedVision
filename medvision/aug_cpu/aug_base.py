#  Copyright (c) 2020. The Medical Image Computing (MIC) Lab, 陶豪毅
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import gc
import time
from typing import List, Tuple, Union, Iterable
from abc import abstractmethod
from copy import deepcopy
import random
from inspect import getframeinfo, stack
from datetime import datetime
import numpy as np

"""
Image:
    numpy array, channel first, shape is zyx order
Classification Label:
    list or numpy array, 1-based
Segmentation Label:
    same with Image, 1-based
Detection Label:
    numpy array, bbox coord is xyz order

"""


class Stage(object):
    """Stage is the smallest component of a pipeline. It is used to do some changes
    on the input data dict 'result'. This stage only supports forward method.

    result: The only object passed through the method, is can be one of the following formats.

    -   A data dictionary . It contains all information used in augmentations.
        e.g. {'img': np.array, 'gt_seg': np.array, ....}

    -   A list of dict contains all 'result' need to handle with.

    TODO replace with Decorators for some utils, such as debug and gc.collect()
    """

    def __init__(self):
        self.name = self.__class__.__name__
        self._debug_ = False
        self._tic_ = None
        self.types = {}  # used for data type checking after changes

    def __repr__(self):
        repr_str = self.__class__.__name__ + '()'
        return repr_str

    def _pre_forward(self, result: dict):
        assert isinstance(result, dict), f'A dict is required but got a {type(result)}!'
        self._debug_ = result.get('_debug_', False)
        self._tic_ = time.time()
        self.try_to_info('_pre_forward')
        self._remember_types(result)
        gc.collect()
        return result

    @abstractmethod
    def _forward(self, result: dict):
        """ do some change to data dict and return it"""
        raise NotImplementedError

    def _post_forward(self, result: dict):
        assert isinstance(result, dict), f'A dict is required but got a {type(result)}!'
        self.try_to_info('_post_forward')
        if 'history' not in result.keys():
            result['history'] = []
        if 'time' not in result.keys():
            result['time'] = []
        _start = datetime.fromtimestamp(self._tic_).strftime('%H:%M:%S.%f')
        result['history'].append(self.name)
        result['time'].append(f"{self.name}-{_start}-{time.time() - self._tic_:.03f}s")
        self._check_types(result)
        gc.collect()
        return result

    def forward(self, result: dict):
        return self._post_forward(self._forward(self._pre_forward(result)))

    def multi_forward(self, result_list: list):
        """ In common, the arg passed through is a list which only contains one data dict """
        # TODO whether to use deepcopy
        return [self.forward(result) for result in result_list]

    def __call__(self, result: Union[dict, List[dict]]):
        """return a same type object as input object"""
        if isinstance(result, dict):
            return self.forward(result)
        elif isinstance(result, list):
            return self.multi_forward(result)
        else:
            raise NotImplementedError

    def _remember_types(self, result: dict):
        self.types = {}
        for k, v in result.items():
            self.types[k] = type(v)
            if k == 'img' and min(v.shape) == 0:
                print(result)
                raise ValueError

    def _check_types(self, result):
        for k, v in result.items():
            if k in self.types.keys():
                assert type(v) == self.types[k],\
                    f"{self.__class__.__name__} ({k}) -> {self.types[k]} ==> {type(v)}"
                if k == 'img' and min(v.shape) == 0:
                    print(result)
                    raise ValueError

    def try_to_info(self, *args):
        if self._debug_:
            caller = getframeinfo(stack()[1][0])
            print(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                  "- {} - line {} :".format(self.__class__.__name__, caller.lineno), end='')
            self.print(*args)

    def info(self, result, *args):
        caller = getframeinfo(stack()[1][0])
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
              "- {} - line {} :".format(self.__class__.__name__, caller.lineno), end='')
        self.print(*args)

    @staticmethod
    def print(*args):
        for arg in args:
            print(arg, end=' ')
        print('')


class OperationStage(Stage):
    """This stage supports both forward(e.g., padding) and backward(e.g., cropping).
    And it will do operation for certain.

    The forward and backward methods are very useful for test time augmentation. For example, Patches inference is
    widely used in medical image field.

    >>> p = AugmentationStage()
    >>> result_origin = {}  # some data
    >>> result_forward = p.forward(result_origin.copy())
    >>> result = p.backward(result_forward)
    >>> for k in ['img', 'other_main_keys']:
    >>>     assert result[k] == result_origin[k]
    """

    def __init__(self):
        super().__init__()
        self.p = 1.0  # probability of doing transformation
        self.latitude = 1.0
        self.dim = None  # image dimension, 2 or 3

        self.array_shape = None  # array shape, c-d-h-w
        self.image_shape = None  # image shape, d-h-w
        self.image_axes = None  # axes except channel
        self.channels = None  # num of image channels

        self.key_name = self.name + '_params'  # dict key name saving in result
        self.params = None  # inner params, both for forward and backward
        self.current_repeat_idx = 0
        self.isForwarding = None  # if forward or backward

    @property
    def canBackward(self):
        return False

    @property
    def repeats(self):
        """ generate num of repeats results(result_list) from one result
        >>> scales = [(32, 32), (64, 64)]
        >>> m = AugmentationStage(arg=scales)
        >>> assert m.repeats == len(scales)
        >>> result = {}  # a dict
        >>> result = m.forward(result)
        >>> assert len(result) == len(scales)
        """
        return 1

    @staticmethod
    def get_range(val_range, bias=0):
        if isinstance(val_range, (int, float)):
            val_range = [- val_range, val_range]
        assert isinstance(val_range, (list, tuple)) and len(val_range) == 2
        return bias + np.random.uniform(*val_range)

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

    @staticmethod
    def fourier_transform(array: np.ndarray):
        transformed = np.fft.fftn(array)
        fshift = np.fft.fftshift(transformed)
        return fshift

    @staticmethod
    def inv_fourier_transform(fshift: np.ndarray):
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifftn(f_ishift)
        return img_back

    def _init_params(self, result):
        """ initial some params """
        self.dim = result['img_dim']
        if 'img' in result.keys():
            self.array_shape = tuple(result['img'].shape)
        else:
            self.array_shape = tuple(result['img_shape'])
        self.image_axes = tuple(range(1, self.dim + 1))
        self.channels = self.array_shape[0]
        self.image_shape = self.array_shape[1:]
        self.params = None

    def _forward_params(self, result):
        """
        Usage:
            must call -> super()._forward_params(result)
            set self.params for forward
            save self.params into result
            apply to img, seg and det
            no need to return anything!
            remember to save changed attrs to result, such as img shape
            remember to clear self.params while using multi forward
        """
        self._init_params(result)

    def _backward_params(self, result):
        """
        Usage:
            must call -> params = super()._backward_params(result)
            if params, do ...
            set self.params
            if self.params is not None, do the following
                apply to img, seg and det
            end if
            no need to return anything!
            remember to save changed attrs to result
            remember to clear self.params while using multi backward
        """
        self._init_params(result)
        params = result.pop(self.key_name, None)
        return params

    def apply_to_img(self, result: dict):
        pass

    def apply_to_cls(self, result: dict):
        pass

    def apply_to_seg(self, result: dict):
        pass

    def apply_to_det(self, result: dict):
        pass

    def __call__(self, result, forward=True):
        if forward:
            # if result is a list, it means it has done multi aug_cuda
            # if self.repeats > 1, it means it need to do multi aug_cuda
            if isinstance(result, list) or self.repeats > 1:
                return self.multi_forward(result)
            else:
                return self.forward(result)
        else:
            # while using backward, it means result has already been a list of dict
            if isinstance(result, list):
                return self.multi_backward(result)
            else:
                return self.backward(result)

    def _forward(self, result: dict):
        # print(self.__class__.__name__, "forward...")
        self.isForwarding = True
        if random.random() <= self.p * self.latitude:
            # print("Doing")
            self.params = None
            self.try_to_info('start _forward_params')
            self._forward_params(result)
            self.try_to_info('start apply_to_img')
            self.apply_to_img(result)
            self.try_to_info('start apply_to_cls')
            self.apply_to_cls(result)
            self.try_to_info('start apply_to_seg')
            self.apply_to_seg(result)
            self.try_to_info('start apply_to_det')
            self.apply_to_det(result)
            self.try_to_info('end')
            # print(self.params)
        return result

    def multi_forward(self, result):
        if self.repeats > 1:
            # generate multi results from the result / or result list
            result = [result] if isinstance(result, dict) else result
            # pay attention to the order of loop, result first and repeats second
            return [self.forward(deepcopy(r)) for r in result for _ in range(self.repeats)]
        else:
            # do transformation on each result with random args
            assert isinstance(result, list) and self.repeats == 1
            return [self.forward(r) for r in result]

    def _pre_backward(self, result: dict):
        time_cost = result['time'].pop()
        last = result['history'].pop()
        assert self.name in time_cost, f'{self.name} {time_cost}'
        assert last == self.name, f'require params to be {self.name}, but the last is {last}'
        self._remember_types(result)
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
        self._check_types(result)
        gc.collect()
        return result

    def backward(self, result):
        return self._post_backward(self._backward(self._pre_backward(result)))

    def multi_backward(self, result: list):
        return [self.backward(r) for r in result]


class AugmentationStage(OperationStage):
    """
    This stage supports easy or hard augmentation with variation of latitude.
    1. latitude equals to 1, it will do augmentation for certain.
    2. latitude equals to 0, it will not do augmentation.

    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.latitude = 1.0

    def setLatitude(self, val):
        self.latitude = val

    def getLatitude(self):
        return self.latitude
