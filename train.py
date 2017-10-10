# -*- coding: utf-8 -*-

"""
QSegNet train routines. (WIP)
"""

import torch
from referit_loader import ReferDataset
from torch.utils.data import DataLoader
from torchvision.transforms import (
    Compose, Scale, CenterCrop, ToPILImage, ToTensor, Normalize)

from utils.transforms import ResizePad, ToNumpy

input_transform = Compose([
    # ToPILImage(),
    # CenterCrop(256),
    # Scale(136),
    ResizePad((300, 300)),
    ToTensor(),
    Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
])

target_transform = Compose([
    ToNumpy(),
    ResizePad((300, 300)),
    ToTensor()
])

refer = ReferDataset(data_root='/mnt/referit_data',
                     dataset='referit',
                     transform=input_transform,
                     annotation_transform=target_transform)
