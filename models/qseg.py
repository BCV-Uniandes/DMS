# -*- coding: utf-8 -*-

"""
Query-based Scene Segmentation (QSeg) Network PyTorch implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from psp import PSPNet
from vlstm import vLSTM, vLSTMCell


class QSegNet(nn.Module):
    def __init__(self, in_size, hid_size, dropout=0.2, num_lstm_layers=2,
                 pretrained=True, batch_first=True):
        self.visual_size = 1024
        self.psp = PSPNet(n_classes=1, pretrained=pretrained)
        self.vlstm = vLSTM(
            vLSTMCell, in_size, hid_size, num_layers=num_lstm_layers,
            batch_first=batch_first, visual_size=self.visual_size)
