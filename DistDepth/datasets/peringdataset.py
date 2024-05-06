# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function

import cv2
import h5py
import numpy as np
import os
import PIL.Image as pil
import skimage.transform

from .peringbase import peringBase

MIN_DEPTH = 0.001
MAX_DEPTH = 10.0


class peringDataset(peringBase):
    def __init__(self, *args, **kwargs):
        super(peringDataset, self).__init__(*args, **kwargs)

        # Normalized intrinsics: The first row is normalize by image_width,
        # the second row is normalized by image_height
        # Reference: https://ksimek.github.io/2013/08/13/intrinsic/.
        self.K = np.array([[0.9375,    0, 0.5, 0],
                           [     0, 1.25, 0.5, 0],
                           [     0,    0,   1, 0],
                           [     0,    0,   0, 1]], dtype=np.float32)

        self.full_res_shape = (640, 480)

    def get_color(self, path, do_flip):
        color = self.loader(path)
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
        return color

    def get_depth(self, path, do_flip):
        depth_gt = cv2.imread(filename=path, flags=cv2.IMREAD_UNCHANGED)
        depth_gt = cv2.resize(
            src=depth_gt,
            dsize=(2 * self.width, 2 * self.height),
            interpolation=cv2.INTER_LANCZOS4
        )
        depth_gt = np.array(depth_gt, dtype=np.float32)
        depth_gt = depth_gt / (2**16 - 1)
        depth_gt = depth_gt * (MAX_DEPTH - MIN_DEPTH) + MIN_DEPTH

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt