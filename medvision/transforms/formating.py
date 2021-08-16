from collections.abc import Sequence

import numpy as np
import torch
import time

from .aug_base import Stage


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


class Collect(Stage):
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
            return [self._forward(r.copy()) for r in results]
        else:
            return [self.backward(r.copy()) for r in results]

    def _forward(self, results):
        _tic_ = time.time()
        data = {}
        img_meta = {}
        for key in self.keys:
            data[key] = to_tensor(results.pop(key))
        for key in ['patches_img']:
            if key in results.keys():
                data[key] = to_tensor(results.pop(key))
        for key in results.keys():
            img_meta[key] = results[key]
        data['img_meta'] = img_meta
        data['img_meta']['history'].append(self.name)
        data['img_meta']['time'].append(f'{self.name}-{time.time() - _tic_:.03f}s')

        return data

    def backward(self, data):
        results = {}
        for key in self.keys:
            results[key] = to_numpy(data[key])
        for key in ['patches_img']:
            if key in data.keys():
                results[key] = to_numpy(data[key])
        for k, v in data['img_meta'].items():
            results[k] = v
        last = results['history'].pop()
        assert last == self.name
        return results
