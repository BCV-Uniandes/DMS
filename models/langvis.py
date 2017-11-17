# -*- coding: utf-8 -*-

"""
Query-based Scene Segmentation (QSegNet) Network PyTorch implementation.
"""

import torch
from sru import SRU
import torch.nn as nn
import torch.nn.functional as F


class LangVisNet(nn.Module):
    def __init__(self, dict_size, emb_size, hid_size, vis_size,
                 num_filters, num_mixed_channels):
        super().__init__()
