# -*- coding: utf-8 -*-

"""
Query-based Scene Segmentation (QSegNet) Network PyTorch implementation.
"""

import torch
import torch.nn as nn

from .psp.pspnet import PSPNet, PSPUpsample
from .vilstm import ViLSTM, ViLSTMCell, ConvViLSTMCell


class QSegNet(nn.Module):
    def __init__(self, in_size, hid_size, vis_size, dropout=0.2,
                 num_vlstm_layers=2, pretrained=True, batch_first=True,
                 psp_size=1024, backend='densenet', dict_size=8054,
                 out_features=512, num_lstm_layers=2):
        super().__init__()
        self.psp = PSPNet(n_classes=1, psp_size=psp_size,
                          pretrained=pretrained, backend=backend,
                          out_features=out_features)
        self.emb = nn.Embedding(dict_size, in_size)
        self.lstm = nn.LSTM(in_size, hid_size, dropout=dropout,
                            batch_first=batch_first)

        # self.vlstm = VILSTM(
        #     VILSTMCell, in_size, hid_size, num_layers=num_lstm_layers,
        #     batch_first=batch_first)

        self.up_1 = PSPUpsample(out_features, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, 64)

        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.LogSoftmax()
        )

    def forward(self, imgs, words):
        psp_features = self.psp(imgs)

        word_emb = self.emb(words)
        out, _ = self.lstm(word_emb)

        # x_x is of size BxLxHxH
        # B: Batch Size
        # L: Phrase length
        # H: Hidden Size
        x_x = torch.matmul(word_emb.unsqueeze(-1), word_emb.unsqueeze(2))
        h_h = torch.matmul(out.unsqueeze(-1), out.unsqueeze(2))
        h_x = torch.matmul(out.unsqueeze(-1), word_emb.unsqueeze(2))

        lang_input = torch.cat(
            [m.unsqueeze(2) for m in (x_x, h_h, h_x)], dim=2)
        # l_t: BxLx1024xHxH
        # l_t = self.lang_conv(lang_input)
        # mask: Bx1024xHxH
        # _, (mask, c) = self.vlstm(l_t, psp_features)

        # p = self.up_1(mask)
        # p = self.drop_2(p)

        # p = self.up_2(p)
        # p = self.drop_2(p)

        # p = self.up_3(p)
        # p = self.drop_2(p)

        # return self.final(p)
        return psp_features, l_t
