# -*- coding: utf-8 -*-

"""
Generic Image Transform utillities.
"""

# import cv2
import torch
import numpy as np
from collections import Iterable

import torch.nn.functional as F
from torch.autograd import Variable


class ResizePad:
    """
    Resize and pad an image to given size.
    """

    def __init__(self, size):
        if not isinstance(size, (int, Iterable)):
            raise TypeError('Got inappropriate size arg: {}'.format(size))

        self.h, self.w = size

    def __call__(self, img):
        h, w = img.size()[-2:]
        scale = min(self.h / h, self.w / w)
        resized_h = int(np.round(h * scale))
        resized_w = int(np.round(w * scale))
        pad_h = int(np.floor(self.h - resized_h) / 2)
        pad_w = int(np.floor(self.w - resized_w) / 2)

        channels = 3 if len(img.size()) > 2 else 1
        num_unsqueeze = 1 + (len(img.size()) <= 2)
        for i in range(num_unsqueeze):
            img = img.unsqueeze(0)

        resized_img = F.upsample(
            Variable(img, volatile=True), size=(resized_h, resized_w),
            mode='bilinear').data
        # resized_img = cv2.resize(img, (resized_w, resized_h))

        new_img = torch.zeros(1, channels, self.h, self.w)
        # if img.ndim > 2:
        # if img.ndim > 2:
        #     new_img = np.zeros(
        #         (self.h, self.w, img.shape[-1]), dtype=resized_img.dtype)
        # else:
        #     resized_img = np.expand_dims(resized_img, -1)
        #     new_img = np.zeros((self.h, self.w, 1), dtype=resized_img.dtype)
        new_img[:, :, pad_h: pad_h + resized_h,
                pad_w: pad_w + resized_w] = resized_img
        return new_img


class CropResize:
    """Remove padding and resize image to its original size."""

    def __call__(self, img, size):
        if not isinstance(size, (int, Iterable)):
            raise TypeError('Got inappropriate size arg: {}'.format(size))
        im_h, im_w = img.data.shape[:2]
        input_h, input_w = size
        scale = max(input_h / im_h, input_w / im_w)
        # scale = torch.Tensor([[input_h / im_h, input_w / im_w]]).max()
        resized_h = int(np.round(im_h * scale))
        # resized_h = torch.round(im_h * scale)
        resized_w = int(np.round(im_w * scale))
        # resized_w = torch.round(im_w * scale)
        crop_h = int(np.floor(resized_h - input_h) / 2)
        # crop_h = torch.floor(resized_h - input_h) // 2
        crop_w = int(np.floor(resized_w - input_w) / 2)
        # crop_w = torch.floor(resized_w - input_w) // 2
        # resized_img = cv2.resize(img, (resized_w, resized_h))
        resized_img = F.upsample(
            img.unsqueeze(0).unsqueeze(0), size=(resized_h, resized_w),
            mode='bilinear')

        resized_img = resized_img.squeeze().unsqueeze(0)

        return resized_img[0, crop_h: crop_h + input_h,
                           crop_w: crop_w + input_w]


class ResizeImage:
    """Resize the largest of the sides of the image to a given size"""
    def __init__(self, size):
        if not isinstance(size, (int, Iterable)):
            raise TypeError('Got inappropriate size arg: {}'.format(size))

        self.size = size

    def __call__(self, img):
        im_h, im_w = img.shape[-2:]
        scale = min(self.size / im_h, self.size / im_w)
        resized_h = int(np.round(im_h * scale))
        resized_w = int(np.round(im_w * scale))
        out = F.upsample(
            Variable(img).unsqueeze(0), size=(resized_h, resized_w),
            mode='bilinear').squeeze().data
        return out


class ToNumpy:
    """Transform an torch.*Tensor to an numpy ndarray."""

    def __call__(self, x):
        return x.numpy()
