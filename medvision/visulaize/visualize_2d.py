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

import numpy as np
# from skimage.morphology import dilation
from scipy.ndimage import grey_dilation
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import cv2


def showImage(image: np.ndarray):
    # https://stackoverflow.com/questions/28816046/displaying-different-images-with-actual-size-in-matplotlib-subplot
    dpi = 300
    fig = plt.figure(figsize=(image.shape[1] / dpi, image.shape[0] / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(image, interpolation="None")
    fig.tight_layout()
    plt.show()


def meshImage(image: np.ndarray):
    # from mpl_toolkits.mplot3d import Axes3D
    assert image.ndim == 2

    y, x = image.shape
    x_values = np.linspace(0, x - 1, x)
    y_values = np.linspace(0, y - 1, y)
    X, Y = np.meshgrid(x_values, y_values)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, image, cmap='jet')
    plt.show()


def getBBox2D(image: np.ndarray,
              bboxes: list,
              labels: list,
              scores: list = None):
    image = image.astype(np.float32)
    if np.max(image) > 1.0 or np.min(image) < 0.0:
        image = Normalize()(image)

    if image.ndim == 2 or image.shape[-1] == 1:
        image = np.dstack([image] * 3)
    font_scale = min(np.sqrt(image.size / 3) / 300, 0.5)
    thickness = min(int((np.sqrt(image.size / 3) - 50) / 100) + 1, 2)
    for i, (b, l) in enumerate(zip(bboxes, labels)):
        x1, y1, x2, y2 = b
        # print(b, l, font_scale, thickness)
        # contains boundary items
        cv2.rectangle(image,
                      (int(x1 - thickness), int(y1 - thickness)),
                      (int(x2 + thickness), int(y2 + thickness)),
                      (1.0, 0, 0), thickness)
        cv2.putText(image, f"{int(l):d}",
                    (int(x1 - thickness), int(y1 - thickness - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (1.0, 0, 0), thickness)
        if scores is not None:
            cv2.putText(image, f"{scores[i] * 100:.0f}",
                        (int(x1 - thickness - 2), int(y2 + thickness + font_scale * 30)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (1.0, 0, 0), thickness)
    return image


def getSeg2D(image: np.ndarray,
             overlay: np.ndarray,
             alpha=0.6):
    """
    get an edge overlaid image to show
    Args:
        image: array shape is h, w, c=1/3
        overlay: array shape is h, w, c=0/1
        alpha:

    Returns:

    """
    image = image.astype(np.float32)
    overlay = overlay.astype(np.float32)
    # overlay = grey_dilation(overlay, 1).astype(np.int32)
    overlay = grey_dilation(overlay, 5) - overlay
    # if np.max(overlay) > 1.0 or np.min(overlay) < 0:
    overlay = Normalize(0, 16)(overlay)  # same label with same color, max label is 16
    if np.max(overlay) > 1.0 or np.min(overlay) < 0:
        overlay = Normalize()(overlay)
    if np.max(image) > 1.0 or np.min(image) < 0.0:
        image = Normalize()(image)
    # print(np.unique(overlay))

    assert np.max(image) <= 1.0 and np.min(image) >= 0.0, f"{np.max(image)}, {np.min(image)}"
    assert np.max(overlay) <= 1.0 and np.min(overlay) >= 0.0, f"{np.max(overlay)}, {np.min(overlay)}"

    if image.ndim == 2 or image.shape[-1] == 1:
        image = np.dstack([image] * 3)
    if overlay.ndim == 3:
        overlay = overlay.squeeze()

    mask = np.dstack([np.zeros_like(overlay)] * 3)
    mask[overlay > 0] = [1, 1, 1]

    overlay = Normalize(0, 1, clip=True)(overlay)

    colormap = plt.cm.gist_rainbow(overlay)[..., :-1] * mask

    out = colormap * alpha + (1 - alpha) * (mask > 0) * image
    out[mask == 0] = image[mask == 0]
    return out


def getZIdxFromSeg(seg: np.ndarray):
    return 1
