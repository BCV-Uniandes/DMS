# -*- coding: utf-8 -*-

"""
Generic Image Transform utillities.
"""

import cv2
import numpy as np
from collections import Iterable


class ResizePad:
    """
    Resize and pad an image to given size.
    """

    def __init__(self, size):
        if not isinstance(size, (int, Iterable)):
            raise TypeError('Got inappropriate size arg: {}'.format(size))

        self.h, self.w = size

    def __call__(self, img):
        h, w = img.shape[:2]
        scale = min(self.h / h, self.w / w)
        resized_h = int(np.round(h * scale))
        resized_w = int(np.round(w * scale))
        pad_h = int(np.floor(self.h - resized_h) / 2)
        pad_w = int(np.floor(self.w - resized_w) / 2)

        resized_img = cv2.resize(img, (resized_w, resized_h))
        if img.ndim > 2:
            new_img = np.zeros(
                (self.h, self.w, img.shape[-1]), dtype=resized_img.dtype)
        else:
            new_img = np.zeros((self.h, self.w), dtype=resized_img.dtype)
        new_img[pad_h: pad_h + resized_h,
                pad_w: pad_w + resized_w, ...] = resized_img
        return new_img


class ToNumpy:
    """Transform an torch.*Tensor to an numpy ndarray."""

    def __call__(self, x):
        return x.numpy()
