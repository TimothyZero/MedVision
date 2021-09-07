from collections.abc import Sequence
from datetime import datetime
import numpy as np
import torch
import time

from .base import CudaAugBase


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(data))
    elif isinstance(data, Sequence) and not isinstance(data, str):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError('type {} cannot be converted to tensor.'.format(
            type(data)))


def to_numpy(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data.numpy()
    elif isinstance(data, np.ndarray):
        return data
    elif isinstance(data, Sequence) and not isinstance(data, str):
        return np.append(data)
    elif isinstance(data, torch.LongTensor):
        return data.numpy()
    # elif isinstance(data, float):
    #     return torch.FloatTensor([data])
    else:
        raise TypeError('type {} cannot be converted to tensor.'.format(
            type(data)))


class CudaCpuToGpu(CudaAugBase):
    def __init__(self, keys=None):
        super().__init__()
        self.keys = keys

    def __repr__(self):
        return self.__class__.__name__ + '(keys={})'.format(self.keys)

    @property
    def canBackward(self):
        return True

    def __call__(self, results, forward=True):
        if isinstance(results, dict):
            results = [results]

        if forward:
            return [self.forward(r) for r in results]
        else:
            return [self.backward(r) for r in results]

    def forward(self, results):
        _tic_ = time.time()
        if self.keys is not None:
            keys = self.keys
        else:
            keys = results.keys()
        for key in keys:
            results[key] = results[key].cuda()

        _start = datetime.fromtimestamp(_tic_).strftime('%H:%M:%S.%f')
        results['history'].append(self.name)
        results['time'].append(f'{self.name}-{_start}-{time.time() - _tic_:.03f}s')
        return results

    def backward(self, results):
        if self.keys is not None:
            keys = self.keys
        else:
            keys = results.keys()
        for key in keys:
            results[key] = results[key].cpu()
        last = results['history'].pop()
        assert last == self.name
        return results


class CudaCollect(CudaAugBase):
    def __init__(self, keys):
        super().__init__()
        self.keys = keys

    def __repr__(self):
        return self.__class__.__name__ + '(keys={})'.format(self.keys)

    @property
    def canBackward(self):
        return True

    def __call__(self, results, forward=True):
        if isinstance(results, dict):
            results = [results]

        if forward:
            return [self.forward(r) for r in results]
        else:
            return [self.backward(r) for r in results]

    def forward(self, results):
        _tic_ = time.time()
        for key in self.keys:
            results[key] = to_tensor(results.pop(key))
        for key in ['patches_img']:
            if key in results.keys():
                results[key] = to_tensor(results.pop(key))
        other_keys = [k for k in results.keys() if k not in self.keys]
        results['img_meta'] = {}
        for key in other_keys:
            results['img_meta'][key] = results.pop(key)
        _start = datetime.fromtimestamp(_tic_).strftime('%H:%M:%S.%f')
        results['img_meta']['history'].append(self.name)
        results['img_meta']['time'].append(f'{self.name}-{_start}-{time.time() - _tic_:.03f}s')

        return results

    def backward(self, data):
        results = {}
        for key in self.keys:
            results[key] = to_tensor(data[key])
        for key in ['patches_img']:
            if key in data.keys():
                results[key] = to_tensor(data[key])
        for k, v in data['img_meta'].items():
            results[k] = v
        last = results['history'].pop()
        assert last == self.name
        return results


class CudaUnCollect(CudaAugBase):
    def __init__(self, keys):
        super().__init__()
        self.keys = keys

    def __repr__(self):
        return self.__class__.__name__ + '(keys={})'.format(self.keys)

    @property
    def canBackward(self):
        return True

    def __call__(self, results, forward=True):
        if isinstance(results, dict):
            results = [results]

        if forward:
            return [self.forward(r) for r in results]
        else:
            return [self.backward(r) for r in results]

    def forward(self, data):
        _tic_ = time.time()
        results = {}
        for key in self.keys:
            results[key] = to_tensor(data[key])
        for key in ['patches_img']:
            if key in data.keys():
                results[key] = to_tensor(data[key])
        for k, v in data['img_meta'].items():
            results[k] = v
        _start = datetime.fromtimestamp(_tic_).strftime('%H:%M:%S.%f')
        results['history'].append(self.name)
        results['time'].append(f'{self.name}-{_start}-{time.time() - _tic_:.03f}s')
        return results

    def backward(self, results):
        for key in self.keys:
            results[key] = to_tensor(results.pop(key))
        for key in ['patches_img']:
            if key in results.keys():
                results[key] = to_tensor(results.pop(key))
        other_keys = [k for k in results.keys() if k not in self.keys]
        results['img_meta'] = {}
        for key in other_keys:
            results['img_meta'][key] = results.pop(key)

        last = results['history'].pop()
        assert last == self.name
        return results