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

from typing import Tuple
from skimage import io
import SimpleITK as sitk
import numpy as np
import os.path as osp
import pickle
import tifffile as tiff


class ImageIO(object):
    # TODO add more efficient handler
    tif_ext = ['.tif', '.tiff']
    cv_ext = ['.png', '.jpg', '.jpeg', '.gif']
    itk_ext = ['.nii.gz', '.nii', '.mhd', '.img']
    npy_ext = ['.npy', '.pkl']

    @staticmethod
    def _test_handler(file_path: str, ext_list: list) -> bool:
        for ext in ext_list:
            if file_path.endswith(ext):
                return True
        return False

    @staticmethod
    def _to_channel_first(data: np.ndarray, dimension: int) -> np.ndarray:
        """ return image with channel first """
        if data.ndim == dimension:  # h,w => c,h,w
            data = data[np.newaxis, ...]
        else:
            if data.shape[-1] == min(data.shape):  # h,w,c => c,h,w
                data = data.transpose((dimension, *range(dimension)))
        assert data.ndim == dimension + 1
        return data

    @staticmethod
    def _to_channel_last(data: np.ndarray) -> Tuple[np.ndarray, int]:
        """ return image with channel last """
        dimension = data.ndim - 1
        if data.shape[0] == min(data.shape):  # c,h,w => h,w,c
            data = data.transpose((*range(1, dimension + 1), 0))
        assert data.shape[-1] == min(data.shape)
        return data, dimension

    @staticmethod
    def is_support_ext(file_path: str) -> bool:
        return max([ImageIO._test_handler(file_path, ext) for ext in
                    [ImageIO.tif_ext, ImageIO.cv_ext, ImageIO.itk_ext, ImageIO.npy_ext]])

    @staticmethod
    def get_proposal_ext(array: np.ndarray) -> str:
        if array.ndim == 2:
            return ImageIO.cv_ext[0]
        elif array.ndim == 3:
            if array.shape[0] in (1, 3):
                return ImageIO.cv_ext[0]
            else:
                return ImageIO.itk_ext[0]
        else:
            return ImageIO.npy_ext[0]

    @staticmethod
    def loadArray(file_path) -> Tuple[np.ndarray, int, tuple, tuple]:
        try:
            file_path = osp.abspath(file_path)
            assert ImageIO.is_support_ext(file_path), file_path

            if ImageIO._test_handler(file_path, ImageIO.tif_ext):
                data = tiff.imread(file_path).squeeze()
                dim = 2
                spacing = (1.0, 1.0)
                origin = (0.0, 0.0)
            elif ImageIO._test_handler(file_path, ImageIO.cv_ext):
                data = np.array(io.imread(file_path))
                dim = 2
                spacing = (1.0, 1.0)
                origin = (0.0, 0.0)
            elif ImageIO._test_handler(file_path, ImageIO.itk_ext):
                data = sitk.ReadImage(file_path)  # xyz order
                dim = data.GetDimension()
                spacing = tuple(reversed(data.GetSpacing()))  # xyz => zyx
                origin = tuple(reversed(data.GetOrigin()))  # xyz => zyx
                data = sitk.GetArrayFromImage(data)  # => zyx
            elif ImageIO._test_handler(file_path, ImageIO.npy_ext):
                # raw shape, should be either (h,w) or (h,w,c) or (d,h,w) or (d,h,w,c)
                loaded_data = np.load(file_path, allow_pickle=True)
                if isinstance(loaded_data, np.ndarray):
                    data = loaded_data
                    dim = sum([i > 16 for i in data.shape])  # assume only channel < 16
                    spacing = (1.0,) * dim
                    origin = (0.0,) * dim
                else:
                    assert isinstance(loaded_data, dict), f'it should be a dict or array, but got a {type(loaded_data)}'
                    data = loaded_data['data']
                    dim = sum([i > 16 for i in data.shape])  # assume only channel < 16
                    spacing = tuple(reversed(loaded_data['spacing']))
                    origin = tuple(reversed(loaded_data['origin']))
            else:
                raise NotImplementedError

            data = ImageIO._to_channel_first(data, dimension=dim)

            assert data.shape[0] == min(data.shape), 'return image should be channel first!'
            assert (dim, data.ndim) in [(2, 3), (3, 4)], "only support 2D or 3D image"

            return data, dim, spacing, origin
        except Exception as e:
            print(file_path)
            raise e

    @staticmethod
    def saveArray(file_path: str,
                  numpyImage: np.ndarray,
                  spacing=(1.0, 1.0, 1.0),
                  origin=(0, 0, 0),
                  orientation=None):
        """

        Args:
            file_path:
            numpyImage: np.array [c,d,h,w]
            spacing: xyz
            origin: xyz
            orientation: xyz

        Returns:

        """
        file_path = osp.abspath(file_path)
        assert ImageIO.is_support_ext(file_path), file_path
        assert numpyImage.ndim in (3, 4), f"only support 2D or 3D image but the image is {numpyImage.ndim}D"
        assert numpyImage.shape[0] == min(numpyImage.shape), f'input image {numpyImage.shape} should be channel first!'

        numpyImage, dim = ImageIO._to_channel_last(numpyImage)
        # spacing and origin should convert into zyx order
        spacing = tuple(reversed(spacing))
        origin = tuple(reversed(origin))
        if orientation is not None:
            orientation = tuple(reversed(orientation))

        if ImageIO._test_handler(file_path, ImageIO.tif_ext):
            assert dim == 2
            assert numpyImage.shape[-1] in (1, 3), numpyImage.shape
            if np.max(numpyImage) <= 1.0:
                numpyImage = (numpyImage * 255).astype(np.uint8)
            if numpyImage.dtype != np.uint8:
                numpyImage = numpyImage.astype(np.uint8)
            numpyImage = np.clip(numpyImage, 0, 255)
            tiff.imsave(file_path, numpyImage)
        elif ImageIO._test_handler(file_path, ImageIO.cv_ext):
            assert dim == 2
            assert numpyImage.shape[-1] in (1, 3), numpyImage.shape
            if np.max(numpyImage) <= 1.0:
                numpyImage = (numpyImage * 255).astype(np.uint8)
            if numpyImage.dtype != np.uint8:
                numpyImage = numpyImage.astype(np.uint8)
            numpyImage = np.clip(numpyImage, 0, 255)
            io.imsave(file_path, numpyImage, check_contrast=False)
        elif ImageIO._test_handler(file_path, ImageIO.itk_ext):
            if numpyImage.ndim == 4:
                assert numpyImage.shape[-1] == 1
                numpyImage = numpyImage[..., 0]
                numpyImage = numpyImage.astype(np.float32)
            else:
                assert numpyImage.shape[-1] in (1, 3)
                numpyImage = numpyImage.astype(np.uint8)
            itkImage = sitk.GetImageFromArray(numpyImage)
            itkImage.SetSpacing(spacing)
            itkImage.SetOrigin(origin)
            if orientation is not None:
                itkImage.SetDirection(orientation)
            sitk.WriteImage(itkImage, file_path)
        elif ImageIO._test_handler(file_path, ImageIO.npy_ext):
            pickle.dump({'data': numpyImage, 'spacing': spacing, 'origin': origin}, open(file_path, 'wb'))
        else:
            pickle.dump({'data': numpyImage, 'spacing': spacing, 'origin': origin}, open(file_path, 'wb'))