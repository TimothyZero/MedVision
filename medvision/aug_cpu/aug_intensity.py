from collections import Iterable
from typing import Union, List, Tuple
import numpy as np
from functools import partial
from scipy import ndimage as ndi
import cv2
import random

from .utils import clipBBoxes
from .base import AugBase


# -------------- channel aug_cuda --------------- #

class RGB2Gray(AugBase):
    def __init__(self):
        super().__init__()
        self.always = True

    @property
    def canBackward(self):
        return True

    def _backward_params(self, result):
        self._init_params(result)
        self.params = True

    def apply_to_img(self, result):
        if self.isForwarding:
            assert self.channels == 3 and self.dim == 2, f"{self.channels} {self.dim}"
            result['img'] = cv2.cvtColor(np.moveaxis(result['img'], 0, -1), cv2.COLOR_RGB2GRAY)[None, ...]
            result['img_shape'] = result['img'].shape
        else:
            assert self.channels == 1
            result['img'] = np.repeat(result['img'], 3, axis=0).astype(np.uint8)
            result['img_shape'] = result['img'].shape
        return result


class Gray2RGB(AugBase):
    def __init__(self):
        super().__init__()
        self.always = True

    @property
    def canBackward(self):
        return True

    def _backward_params(self, result):
        self._init_params(result)
        self.params = True

    def apply_to_img(self, result):
        if self.isForwarding:
            assert self.channels == 1
            result['img'] = np.repeat(result['img'], 3, axis=0).astype(np.uint8)
            result['img_shape'] = result['img'].shape
        else:
            assert self.channels == 3 and self.dim == 2
            result['img'] = result['img'][[0], ...]
            result['img_shape'] = result['img'].shape
        return result


class ChannelSelect(AugBase):
    def __init__(self, index: (list, tuple, int)):
        super().__init__()
        self.always = True
        if isinstance(index, (int, float)):
            index = [int(index)]
        self.index = index
        assert isinstance(self.index, (list, tuple))

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(channel_index={})'.format(self.index)
        return repr_str

    @property
    def canBackward(self):
        return True

    def _forward_params(self, result):
        self._init_params(result)
        self.params = tuple([self.index, self.channels])
        result[self.key_name] = self.params

    def _backward_params(self, result):
        self._init_params(result)
        params = result.pop(self.key_name, None)
        if params:
            self.params = params

    def apply_to_img(self, result):
        if self.isForwarding:
            index, _ = self.params
            result['img'] = result['img'].take(indices=index, axis=0)
            result['img_shape'] = result['img'].shape
        else:
            _, channels = self.params
            result['img'] = np.repeat(result['img'], channels, axis=0)
            result['img_shape'] = result['img'].shape
        return result


class AnnotationMap(AugBase):
    def __init__(self, mapping: dict):
        super().__init__()
        self.always = True
        self.mapping = mapping
        assert isinstance(self.mapping, dict)
        assert all([isinstance(i, int) for i in self.mapping.keys()])
        assert all([isinstance(i, int) for i in self.mapping.values()])

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(mapping={})'.format(self.mapping)
        return repr_str

    @property
    def canBackward(self):
        flag = len(set(self.mapping.values())) == len(self.mapping.values())
        flag = flag and len(set(self.mapping.keys())) == len(self.mapping.keys())
        return flag

    def _forward_params(self, result):
        self._init_params(result)
        self.params = self.mapping.copy()
        result[self.key_name] = self.params

    def _backward_params(self, result):
        self._init_params(result)
        params = result.pop(self.key_name, None)
        self.params = dict((v, k) for k, v in params.items())

    def apply_to_cls(self, result):
        for key in result.get('cls_fields', []):
            for prev, curr in self.params.items():
                result[key] = curr if result[key] == prev else result[key]
        return result

    def apply_to_seg(self, result):
        for key in result.get('seg_fields', []):
            for prev, curr in self.params.items():
                result[key] = np.where(result[key] == prev, curr, result[key])
        return result

    def apply_to_det(self, result):
        for key in result.get('det_fields', []):
            for prev, curr in self.params.items():
                bboxes_labels = result[key][:, 2 * self.dim]
                bboxes_labels = np.where(bboxes_labels == prev, curr, bboxes_labels)
                result[key][:, 2 * self.dim] = bboxes_labels


# -------------- normalization --------------- #


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
        self._init_params(result)
        params = result.pop(self.key_name, None)
        if params is not None:
            # [(-1, -1, -1), (1/128, 1/128, 1/128)]
            r_mean = - np.array(params[0]) / np.array(params[1])
            r_std = 1 / np.array(params[1])
            self.params = [tuple(r_mean), tuple(r_std)]

    def apply_to_img(self, result):
        mean, std = self.params
        assert self.channels == len(mean) == len(std), f"channels = {self.channels}"
        assert result['img'].shape == result['img_shape']
        expand = (slice(None),) + (None,) * self.dim
        result['img'] = (result['img'] - np.array(mean, dtype=np.float32)[expand]) / np.array(std, dtype=np.float32)[
            expand]
        if self.clip and self.isForwarding:
            result['img'] = np.clip(result['img'], -1.0, 1.0)


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
            img = result['img'].astype(np.float32)
            means, stds = self.params
            assert self.channels == len(means[0]) == len(stds[0]), f"channels = {self.channels}, it should be 1"
            assert img.shape == result['img_shape']
            expand = (slice(None),) + (None,) * self.dim
            imgs = [(img - np.array(mean, dtype=np.float32)[expand]) / np.array(std, dtype=np.float32)[expand]
                    for mean, std in zip(means, stds)]
            img = np.concatenate(imgs, axis=0)
            if self.clip and self.isForwarding:
                img = np.clip(img, -1.0, 1.0)
            result['img'] = img
            result['img_shape'] = img.shape
        else:
            img = result['img'].astype(np.float32)
            mean, std = self.params
            assert self.channels == len(mean) == len(std), f"channels={self.channels}, mean={mean}"
            assert img.shape == result['img_shape']
            expand = (slice(None),) + (None,) * self.dim
            img = (img - np.array(mean, dtype=np.float32)[expand]) / np.array(std, dtype=np.float32)[expand]
            img = np.mean(img, axis=0, keepdims=True)
            result['img'] = img
            result['img_shape'] = img.shape


class AutoNormalize(AugBase):
    """Normalize the image to [-1.0, 1.0].
    """

    def __init__(self, method='norm', clip=False):
        super().__init__()
        self.always = True
        self.method = method
        self.clip = clip
        assert method in ['norm', 'minmax'], "method is one of ['norm', 'minmax']"

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(method={}, clip={})'.format(self.method, self.clip)
        return repr_str

    @property
    def canBackward(self):
        return True

    def _forward_params(self, result):
        self._init_params(result)
        if self.method == 'norm':
            mean = np.mean(result['img'], axis=self.image_axes)
            std = np.std(result['img'], axis=self.image_axes)
        else:
            M = np.max(result['img'], axis=self.image_axes)
            m = np.min(result['img'], axis=self.image_axes)
            mean = (M + m) / 2
            std = (M - m) / 2
        if not isinstance(mean, Iterable):
            mean = [mean]
            std = [std]
        self.params = [tuple(mean), tuple(std)]
        result[self.key_name] = self.params

    def _backward_params(self, result):
        self._init_params(result)
        params = result.pop(self.key_name, None)
        if params is not None:
            r_mean = - np.array(params[0]) / np.array(params[1])
            r_std = 1 / np.array(params[1])
            self.params = [tuple(r_mean), tuple(r_std)]

    def apply_to_img(self, result):
        img = result['img'].astype(np.float32)
        mean, std = self.params
        assert self.channels == len(mean) == len(std), f"channels = {self.channels}, mean = len({len(mean)})"
        assert img.shape == result['img_shape']
        expand = (slice(None),) + (None,) * self.dim
        img = (img - np.array(mean, dtype=np.float32)[expand]) / np.array(std, dtype=np.float32)[expand]
        if self.clip and self.isForwarding:
            img = np.clip(img, -1.0, 1.0)
        result['img'] = img


# ------------- intensity ---------------- #

class RandomGamma(AugBase):
    """
    support segmentation, detection and classification tasks
    support 2D and 3D images
    """

    def __init__(self, p, gamma):
        super().__init__()
        self.p = p
        self.gamma = gamma

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(p={}, gamma={})'.format(self.p, self.gamma)
        return repr_str

    @property
    def canBackward(self):
        return True

    def _forward_params(self, result):
        self._init_params(result)
        if isinstance(self.gamma, (list, tuple)):
            # assert len(self.gamma) == self.channels, "len(gamma) must equals to image channels"
            assert len(self.gamma) == 2 and self.gamma[0] < self.gamma[1], \
                "gamma is [min, max] format or just a number"
        gamma = tuple([self.get_range(self.gamma, 1)] * self.channels)
        self.params = gamma
        result[self.key_name] = self.params

    def _backward_params(self, result):
        self._init_params(result)
        params = result.pop(self.key_name, None)
        if params is not None:
            self.params = tuple([1 / p for p in params])

    def apply_to_img(self, result):
        image = result['img']
        new_image = np.zeros_like(image)
        for c in range(self.channels):
            c_image = image[c]
            temp_min, temp_max = np.min(c_image) - 1e-5, np.max(c_image) + 1e-5
            c_image = (c_image - temp_min) / (temp_max - temp_min)
            c_image = np.power(c_image, self.params[c])
            new_image[c] = c_image * (temp_max - temp_min) + temp_min
        result['img'] = new_image


class RandomBlur(AugBase):
    """
    support segmentation, detection and classification tasks
    support 2D and 3D images
    """

    def __init__(self, p, sigma):
        super().__init__()
        self.p = p
        self.sigma = sigma

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(p={}, sigma={})'.format(self.p, self.sigma)
        return repr_str

    @property
    def canBackward(self):
        return True

    def _forward_params(self, result):
        self._init_params(result)
        if isinstance(self.sigma, (list, tuple)):
            # assert len(self.sigma) == self.channels, "len(sigma_std) must equals to image channels"
            # sigma = [sigma * random.random() for sigma in self.sigma]
            assert len(self.sigma) == 2 and self.sigma[0] <= self.sigma[1]
            sigma = [self.get_range(self.sigma)] * self.channels
        else:
            sigma = [self.sigma * random.random()] * self.channels
        self.params = sigma
        result[self.key_name] = self.params

    def _backward_params(self, result):
        super()._backward_params(result)
        self.params = [True]

    def apply_to_img(self, result):
        if self.isForwarding:
            image = result['img']
            new_image = np.zeros_like(image)
            for c in range(self.channels):
                c_image = image[c]
                c_image = ndi.gaussian_filter(c_image, sigma=self.params[c])
                new_image[c] = c_image
            result['img'] = new_image


class RandomNoise(AugBase):
    def __init__(self,
                 p: float,
                 method: str = 'uniform',
                 mean: float = 0,
                 std: float = 0.1):
        super().__init__()
        self.supported = ['uniform', 'normal']
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
        if self.method == 'uniform':
            noise = ((np.random.rand(*self.array_shape) - 0.5) / 0.5) * self.std + self.mean
        else:
            noise = np.random.randn(*self.array_shape) * self.std + self.mean
        self.params = noise.astype(np.float32)
        result[self.key_name] = self.params

    def _backward_params(self, result):
        self._init_params(result)
        params = result.pop(self.key_name, None)
        if params is not None:
            self.params = -params

    def apply_to_img(self, result):
        result['img'] = result['img'] + self.params


class RandomSpike(AugBase):
    def __init__(self,
                 p,
                 num_spikes: Union[int, Tuple[int, int]] = 1,
                 intensity: Union[float, Tuple[float, float]] = (0.5, 1)
                 ):
        super().__init__()
        self.p = p
        if isinstance(num_spikes, int):
            num_spikes = (1, num_spikes)
        self.num_spikes = num_spikes
        self.intensity = intensity

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(num_spikes={})'.format(self.num_spikes)
        return repr_str

    @property
    def canBackward(self):
        return True

    def _forward_params(self, result):
        self._init_params(result)
        num_spikes_param = int(self.get_range(self.num_spikes))
        intensity_param = self.get_range(self.intensity)
        spikes_positions = np.random.rand(num_spikes_param, self.dim)
        self.params = spikes_positions.tolist(), intensity_param
        result[self.key_name] = self.params

    def _backward_params(self, result):
        self._init_params(result)
        params = result.pop(self.key_name, None)
        if params is not None:
            spikes_positions, intensity_param = params
            self.params = spikes_positions, - intensity_param

    def apply_to_img(self, result):
        image = result['img']
        spikes_positions, intensity = self.params

        transformed_result = []
        for c in image:
            spectrum = self.fourier_transform(c)
            if intensity >= 1 and not self.isForwarding:
                tmp = spectrum.max() / intensity
            else:
                tmp = spectrum.max()
            spikes_positions = np.array(spikes_positions)
            shape = np.array(self.image_shape)
            mid_shape = shape // 2
            indices = np.floor(spikes_positions * shape).astype(int)
            for index in indices:
                diff = index - mid_shape
                idx = mid_shape + diff
                spectrum[tuple(idx)] += tmp * intensity
                # If we wanted to add a pure cosine, we should add spikes to both
                # sides of k-space. However, having only one is a better
                # representation og the actual cause of the artifact in real
                # scans. Therefore the next two lines have been removed.
                # #i, j, k = mid_shape - diff
                # #spectrum[i, j, k] = spectrum.max() * intensity_factor
            cc = np.real(self.inv_fourier_transform(spectrum))
            transformed_result.append(cc)
        result['img'] = np.stack(transformed_result, axis=0)


class RandomBiasField(AugBase):
    def __init__(self, p, coefficients):
        super().__init__()
        self.p = p
        self.coefficients = coefficients
        self.order = 1

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(coefficients={})'.format(self.coefficients)
        return repr_str

    @property
    def canBackward(self):
        return True

    def _forward_params(self, result):
        self._init_params(result)
        random_coefficients = []
        if self.dim == 3:
            for x_order in range(0, self.order + 1):
                for y_order in range(0, self.order + 1 - x_order):
                    for _ in range(0, self.order + 1 - (x_order + y_order)):
                        number = self.get_range(self.coefficients)
                        random_coefficients.append(number)
        else:
            for x_order in range(0, self.order + 1):
                for y_order in range(0, self.order + 1 - x_order):
                    number = self.get_range(self.coefficients)
                    random_coefficients.append(number)
        random_coefficients = np.array(random_coefficients)
        self.params = random_coefficients
        result[self.key_name] = random_coefficients.tolist()

    def _backward_params(self, result):
        self._init_params(result)
        params = result.pop(self.key_name, None)
        if params is not None:
            self.params = params

    def apply_to_img(self, result):
        image = result['img']
        transformed_result = []
        for component in image:
            half_shape = np.array(component.shape) / 2

            ranges = [np.arange(-n, n) for n in half_shape]

            bias_field = np.zeros(component.shape)

            if self.dim == 3:
                x_mesh, y_mesh, z_mesh = np.asarray(np.meshgrid(*ranges, indexing='ij'))

                x_mesh /= x_mesh.max()
                y_mesh /= y_mesh.max()
                z_mesh /= z_mesh.max()
                i = 0
                for x_order in range(self.order + 1):
                    for y_order in range(self.order + 1 - x_order):
                        for z_order in range(self.order + 1 - (x_order + y_order)):
                            random_coefficient = self.params[i]
                            new_map = (
                                    random_coefficient
                                    * x_mesh ** x_order
                                    * y_mesh ** y_order
                                    * z_mesh ** z_order
                            )
                            bias_field += new_map
                            i += 1
            else:
                x_mesh, y_mesh = np.asarray(np.meshgrid(*ranges, indexing='ij'))

                x_mesh /= x_mesh.max()
                y_mesh /= y_mesh.max()
                i = 0
                for x_order in range(self.order + 1):
                    for y_order in range(self.order + 1 - x_order):
                        random_coefficient = self.params[i]
                        new_map = (
                                random_coefficient
                                * x_mesh ** x_order
                                * y_mesh ** y_order
                        )
                        bias_field += new_map
                        i += 1
            bias_field = np.exp(bias_field).astype(np.float32)
            bias_field = bias_field / np.max(bias_field)
            if self.isForwarding:
                component = component * bias_field
            else:
                component = component / bias_field
            transformed_result.append(component)
        result['img'] = np.stack(transformed_result, axis=0)


class RandomCutout(AugBase):
    FUSION = {
        'mean': np.mean,
        'min': np.min,
        'max': np.max
    }

    def __init__(self, p,
                 num_holes: int,
                 size: int,
                 apply_to: Union[tuple, list] = (),
                 fill='mean'):
        super(RandomCutout, self).__init__()
        self.p = p
        self.num_holes = num_holes
        self.size = size
        self.apply_to = apply_to
        if isinstance(fill, (int, float)):
            self.fusion_fun = partial(lambda a, constant: constant, constant=fill)
        else:
            self.fusion_fun = RandomCutout.FUSION[str(fill)]

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(p={}, num_holes={}, size={}, apply_to={})' \
            .format(self.p, self.num_holes, self.size, self.apply_to)
        return repr_str

    @property
    def canBackward(self):
        return True

    def _forward_params(self, result):
        self._init_params(result)
        ctr = np.random.randint(0, self.image_shape[::-1], size=(self.num_holes, self.dim))  # xyz
        bboxes = np.concatenate([ctr, ctr + self.size], axis=1)
        bboxes = clipBBoxes(self.dim, bboxes, self.image_shape)
        self.params = bboxes
        result[self.key_name] = self.params

    def _backward_params(self, result):
        self._init_params(result)
        self.params = True

    def apply_to_img(self, result: dict):
        # print(result['img'].shape)
        if self.isForwarding:
            mask = np.zeros_like(result['img'])[[0], ...]
            for hole in self.params:
                slices = (slice(None),) + tuple(map(slice, hole[:self.dim][::-1], hole[-self.dim:][::-1]))
                mean_val = self.fusion_fun(result['img'][slices])
                result['img'][slices] = mean_val
                mask[slices] = 1.0
            result['cutout_mask'] = mask
            result['seg_fields'].append('cutout_mask')

    def apply_to_seg(self, result: dict):
        if self.isForwarding:
            for key in result['seg_fields']:
                if key in self.apply_to:
                    for hole in self.params:
                        slices = (slice(None),) + tuple(map(slice, hole[:self.dim][::-1], hole[-self.dim:][::-1]))
                        result[key][slices] = 0


class ForegroundCutout(RandomCutout):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _forward_params(self, result):
        self._init_params(result)
        if 'gt_seg_skeleton' in result.keys():
            foreground = result['gt_seg_skeleton']
        else:
            foreground = result['gt_seg']
        points = np.argwhere(foreground[0] > 0)  # zyx
        if len(points):
            ctr = points[np.random.choice(np.arange(len(points)), self.num_holes)][:, ::-1]  # xyz
            bboxes = np.concatenate([ctr, ctr + self.size], axis=1)
            bboxes = clipBBoxes(self.dim, bboxes, self.image_shape)
            self.params = bboxes
            result[self.key_name] = self.params
        else:
            RandomCutout._forward_params(self, result)
