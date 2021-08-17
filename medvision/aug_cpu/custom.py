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

from skimage import measure, morphology
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from .aug_base import OperationStage
from .aug_intensity import RandomGamma


class MultiGammaEns(OperationStage):
    def __init__(self, gammas: list):
        super().__init__()
        assert min(gammas) > -1.0 and max(gammas) < 1.0
        self.gammas = gammas

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(p={}, gammas={})'.format(self.p, self.gammas)
        return repr_str

    @property
    def canBackward(self):
        return True

    def _forward_params(self, result):
        super()._forward_params(result)
        gamma = [tuple([gamma + 1] * self.channels) for gamma in self.gammas]
        self.params = gamma
        result[self.key_name] = self.params

    def _backward_params(self, result):
        params = super()._backward_params(result)
        if params is not None:
            self.params = [tuple([1 / p for p in param]) for param in params]

    def apply_to_img(self, result):
        if self.isForwarding:
            assert self.channels == 1
            image = result['img']
            imgs = []
            for m in range(len(self.gammas)):
                new_image = np.zeros_like(image)
                for c in range(self.channels):
                    c_image = image[c]
                    temp_min, temp_max = np.min(c_image) - 1e-5, np.max(c_image) + 1e-5
                    c_image = (c_image - temp_min) / (temp_max - temp_min)
                    c_image = np.power(c_image, self.params[m][c])
                    new_image[c] = c_image * (temp_max - temp_min) + temp_min
                imgs.append(new_image)
            imgs = np.concatenate(imgs, axis=self.channel_axis)
            result['img'] = imgs
            result['img_shape'] = imgs.shape
            # print(imgs.shape)
        else:
            # print(self.params)
            imgs = []
            for m in range(len(self.params)):
                start = m * self.channels // len(self.params)
                stop = (m + 1) * self.channels // len(self.params)
                image = result['img'][start:stop]
                new_image = np.zeros_like(image)
                # print(image.shape)
                for c in range(self.channels // len(self.params)):
                    c_image = image[c]
                    temp_min, temp_max = np.min(c_image) - 1e-5, np.max(c_image) + 1e-5
                    c_image = (c_image - temp_min) / (temp_max - temp_min)
                    c_image = np.power(c_image, self.params[m][c])
                    new_image[c] = c_image * (temp_max - temp_min) + temp_min
                imgs.append(new_image)
            # img = result['img'].astype(np.float32)
            # assert self.channels == len(mean) == len(std), f"channels = {self.channels}"
            # assert img.shape == result['img_shape']
            imgs = np.concatenate(imgs, axis=self.channel_axis)
            img = np.mean(imgs, axis=-1, keepdims=True)
            # print(img.shape)
            result['img'] = img
            result['img_shape'] = img.shape


# class LungMask(object):
#     def __init__(self, fill_lung_structures=True, debug=False):
#         self.fill_lung_structures = fill_lung_structures
#         self.debug = debug
#
#     @staticmethod
#     def largest_label_volume(im, bg=-1):
#         vals, counts = np.unique(im, return_counts=True)
#
#         counts = counts[vals != bg]
#         vals = vals[vals != bg]
#
#         # print(vals)
#         # print(counts)
#
#         if len(counts) > 0:
#             return vals[np.argmax(counts)]
#         else:
#             return None
#
#     def _apply_to_img(self, result):
#         numpyImage = result['img']
#         d, h, w = numpyImage.shape
#         # if debug:
#         #     vtkShow(numpyImage)
#
#         # => outer/inner air = 1, body = 0
#         binary_image = np.array(numpyImage < -320, dtype=np.float)
#         # binary_image = morphology.binary_closing(binary_image, np.ones([2, 2, 2]))
#         # binary_image = morphology.binary_opening(binary_image, np.ones([2, 2, 2]))
#
#         # if debug:
#         #     plt.imshow(binary_image[d // 2, :, :], 'gray')
#         #     plt.title("binary image")
#         #     plt.show()
#         #     vtkShow(binary_image)
#
#         # Fill the air around the person
#         # => inner air = 1, body/outer air = 0
#         labels = measure.label(binary_image, connectivity=1)
#         for i in range(9):
#             c = f'{i:03b}'
#             background_label = labels[(d - 1) * int(c[0]), (h - 1) * int(c[1]), (w - 1) * int(c[2])]
#             binary_image[background_label == labels] = 0
#             # plt.imshow(binary_image[d // 2, :, :], 'gray')
#             # plt.title(i)
#             # plt.show()
#
#         if self.debug:
#             plt.subplot(2, 2, 1)
#             plt.imshow(numpyImage[d // 2, :, :], 'gray')
#             plt.title("input image")
#             plt.subplot(2, 2, 2)
#             plt.imshow(Normalize()(labels[d // 2, :, :]), 'jet')
#             plt.title(f'labeled outer/inner air')
#             plt.subplot(2, 2, 3)
#             plt.imshow(binary_image[d // 2, :, :], 'gray')
#             plt.title("inner air")
#             plt.show()
#
#         # Method of filling the lung structures (that is superior to something like
#         # morphological closing)
#         if self.fill_lung_structures:
#             # For every slice we determine the largest solid structure
#             for i, axial_slice in enumerate(binary_image):
#                 axial_slice = axial_slice + 1
#                 labeling = measure.label(axial_slice)
#                 l_max = self.largest_label_volume(labeling, bg=0)
#
#                 if l_max is not None:  # This slice contains some lung
#                     binary_image[i][labeling != l_max] = 1
#
#         # Remove other air pockets insided body
#
#         # if debug:
#         #     plt.imshow(Normalize()(labels[d // 2, :, :]), 'jet')
#         #     plt.title(f'labeled in {np.max(labels.flatten())} classes after fill outer air')
#         #     plt.show()
#         labels = measure.label(binary_image, background=0, connectivity=1)
#         l_max = self.largest_label_volume(labels, bg=0)
#         if l_max is not None:  # There are air pockets
#             binary_image[labels != l_max] = 0
#
#         return binary_image
#
#     # def segment_lung_mask(numpyImage, fill_lung_structures=True, debug=False):
#     #     d, h, w = numpyImage.shape
#     #     if debug:
#     #         plt.imshow(numpyImage[d // 2, :, :])
#     #         plt.title("input")
#     #         plt.show()
#     #
#     #     # outer/inner air = 0, body = 1
#     #     binary_image = np.array(numpyImage > -320, dtype=np.int8)
#     #     binary_image = morphology.binary_closing(binary_image, np.ones([2, 2, 2]))
#     #
#     #     # outer/inner air = 1, body = 2
#     #     binary_image = binary_image + 1
#     #     if debug:
#     #         plt.imshow(binary_image[d // 2, :, :])
#     #         plt.title("binary")
#     #         plt.show()
#     #
#     #         # vtkShow(binary_image)
#     #
#     #     labels = measure.label(binary_image)
#     #     # Pick the pixel in the very corner to determine which label is air.
#     #     #   Improvement: Pick multiple background labels from around the patient
#     #     #   More resistant to "trays" on which the patient lays cutting the air
#     #     #   around the person in half
#     #     background_label = labels[0, 0, 0]
#     #
#     #     # Fill the air around the person
#     #     binary_image[background_label == labels] = 2
#     #     if debug:
#     #         plt.imshow(labels[d // 2, :, :] / np.max(labels[d // 2, :, :], axis=(0, 1)), 'jet')
#     #         plt.title(f'labeled in {np.max(labels.flatten())} classes')
#     #         plt.show()
#     #
#     #     # Method of filling the lung structures (that is superior to something like
#     #     # morphological closing)
#     #     if fill_lung_structures:
#     #         # For every slice we determine the largest solid structure
#     #         for i, axial_slice in enumerate(binary_image):
#     #             axial_slice = axial_slice - 1
#     #             labeling = measure.label(axial_slice)
#     #             l_max = largest_label_volume(labeling, bg=0)
#     #
#     #             if l_max is not None:  # This slice contains some lung
#     #                 binary_image[i][labeling != l_max] = 1
#     #
#     #     binary_image -= 1  # Make the image actual binary
#     #     binary_image = 1 - binary_image  # Invert it, lungs are now 1
#     #
#     #     # Remove other air pockets insided body
#     #     labels = measure.label(binary_image, background=0)
#     #     l_max = largest_label_volume(labels, bg=0)
#     #     if l_max is not None:  # There are air pockets
#     #         binary_image[labels != l_max] = 0
#     #
#     #     if debug:
#     #         vtkShow(binary_image)
#     #     return binary_image
