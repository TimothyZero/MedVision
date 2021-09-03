import os
import time
import numpy as np
from PIL import Image
import cv2
import torch
from matplotlib.colors import Normalize
import random
from imageio import mimsave

from .base import CudaAugBase
from ..visulaize import getSeg2D, getBBox2D


def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    # https://stackoverflow.com/questions/35180764/opencv-python-image-too-big-to-display
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


class CudaDisplay(CudaAugBase):
    def __init__(self):
        super(CudaDisplay, self).__init__()
        self.p = 1

    def _forward(self, result, tab=1):
        if tab == 1:
            print("")
        for k, v in result.items():
            if isinstance(v, torch.Tensor):
                v = v.cpu().numpy()
                k += f"(Tensor:{v.dtype})"
            if isinstance(v, np.ndarray):
                k += f"(Array:{v.dtype})"
                if v.ndim >= 3 or v.size > 64:
                    print("-" * tab, k, ': shape=', v.shape, 'range=', (np.min(v), np.max(v)))
                else:
                    print("-" * tab, k, ':')
                    print(v)
            elif isinstance(v, dict):
                print("-" * tab, k, ':')
                self._forward(v, tab + 2)
            else:
                print("-" * tab, k, ':', v)
        if tab == 1:
            print("")
        return result

    def forward(self, result: dict):
        return self._forward(result)


class CudaViewer(CudaAugBase):
    """
    TODO: multi modality visualization
    support transposed tensor and numpy array
    used in dataset pipeline, not after loader
    """
    def __init__(self, save_dir=None, p=1.0, advance=False):
        super().__init__()
        self.save_dir = save_dir
        self.p = p
        self.advance = advance
        self.idx = 0
        self.dim = None

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += "(save_dir={}, p={})".format(self.save_dir, self.p)
        return repr_str

    def _forward(self, result: dict):
        assert 'img' in result.keys()
        if random.random() > self.p:
            return

        if 'img_meta' in result.keys():
            self.dim = result['img_meta']['img_dim']
        else:
            self.dim = result['img_dim']
        if self.dim == 2:
            self.__view2D(result)
        elif self.dim == 3:
            self.__view3D(result)
        return result

    def forward(self, result: dict):
        return self._forward(result)

    @staticmethod
    def force_numpy(result, key):
        data = result[key]
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
            return data.copy()
        elif isinstance(data, np.ndarray):
            return data.copy()
        else:
            return data.copy()

    def __view2D(self, result):
        raw_image = self.force_numpy(result, 'img')
        raw_image = raw_image * 0.5 + 0.5  # [-1, 1] -> [0, 1]

        if not (np.max(raw_image) <= 1.0 and np.min(raw_image) >= 0):
            print('\033[31m{}-Warning: Normalization to [-1, 1] is recommended!\033[0m'.format(self.__class__.__name__))
            raw_image = Normalize()(raw_image)

        for c in range(raw_image.shape[0]):
            print(f"Select No.{c} channel of image to show ...")
            image = raw_image[c]
            image = np.stack([image] * 3, axis=-1).squeeze()

            # draw bboxes if available
            if 'gt_det' in result.keys():
                det = self.force_numpy(result, 'gt_det')
                bboxes = det[:, :4]
                labels = det[:, 4]
                scores = det[:, 5]
                image = getBBox2D(image, bboxes, labels, scores)

            if 'gt_seg' in result.keys():
                seg = self.force_numpy(result, 'gt_seg')
                seg = seg[0]
                image = getSeg2D(image, seg)

            image = (image * 255).astype(np.uint8)
            if self.save_dir:
                try:
                    if 'img_meta' in result.keys():
                        filename = result['img_meta']['filename']
                    else:
                        filename = result['filename']
                except Exception:
                    filename = self.idx
                cv2.imwrite(os.path.join(self.save_dir, f"{filename}_idx{self.idx}.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                self.idx += 1
            else:
                while True:
                    if np.max(image.shape) > 1024:
                        image = ResizeWithAspectRatio(image, width=1024, height=1024)

                    cv2.imshow("Normalized Image", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                    if cv2.waitKey(100) & 0xFF == 27:  # exit while pressing ESC
                        break
                    if cv2.getWindowProperty('Normalized Image', cv2.WND_PROP_VISIBLE) < 1:  # closing window
                        break
                cv2.destroyAllWindows()

    def __view3D(self, result):
        raw_image = self.force_numpy(result, 'img')
        raw_image = raw_image * 0.5 + 0.5  # [-1, 1] -> [0, 1]

        if not (np.max(raw_image) <= 1.0 and np.min(raw_image) >= 0):
            print('\033[31m{}-Warning: Normalization to [-1, 1] is recommended!\033[0m'.format(self.__class__.__name__))
            raw_image = Normalize()(raw_image)

        for c in range(raw_image.shape[0]):
            print(f"Select No.{c} channel of image to show ...")
            image = raw_image[c]
            image = np.stack([image] * 3, axis=-1).squeeze()

            if 'gt_det' in result.keys():
                det = self.force_numpy(result, 'gt_det')
                bboxes = det[:, :6]
                labels = det[:, 6]
                scores = det[:, 7]
                for i in range(image.shape[0]):  # z direction
                    tmp_bboxes = []
                    tmp_labels = []
                    tmp_scores = []
                    for idx, bbox in enumerate(bboxes):
                        if bbox[2] <= i <= bbox[5]:
                            tmp_bboxes.append(bbox[[0, 1, 3, 4]])
                            tmp_labels.append(labels[idx])
                            tmp_scores.append(scores[idx])
                    if len(tmp_bboxes):
                        im = getBBox2D(image[i, ...], tmp_bboxes, tmp_labels, tmp_scores)
                        image[i, ...] = im

            if 'gt_seg' in result.keys():
                ori_shape = list(image.shape)
                print("Only segmentation channel 0 is showed")
                seg = self.force_numpy(result, 'gt_seg')
                seg = seg[0]
                seg = np.reshape(seg, (-1, seg.shape[2], 1))
                image = np.reshape(image, (-1, image.shape[2], 3))
                image = getSeg2D(image, seg)
                image = np.reshape(image, ori_shape)

            if self.save_dir:
                """ save a gif"""
                try:
                    if 'img_meta' in result.keys():
                        filename = result['img_meta']['filename']
                    else:
                        filename = result['filename']
                except Exception:
                    filename = self.idx
                images = []
                for i in range(len(image)):
                    im = image[i, ...] * 255
                    im = Image.fromarray(im.astype(np.uint8))
                    images.append(im)
                mimsave(os.path.join(self.save_dir, f"{filename}_{c}_idx{self.idx}_imageio.gif"), images)
                self.idx += 1

            else:
                """ show animate gif"""
                images = [cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR) for img in image]
                i = 0
                while True:
                    if np.max(images[i].shape) > 1024:
                        resized = ResizeWithAspectRatio(images[i], width=1024, height=1024)
                    elif np.max(images[i].shape) < 512:
                        resized = ResizeWithAspectRatio(images[i], width=512, height=512)
                    else:
                        resized = images[i]

                    cv2.imshow("gif", resized)
                    if cv2.waitKey(100) & 0xFF == 27:  # exit while pressing ESC
                        break
                    if cv2.getWindowProperty('gif', cv2.WND_PROP_VISIBLE) < 1:  # exit while closing window
                        break
                    i = (i + 1) % len(images)
                    time.sleep(0.05)
                cv2.destroyAllWindows()