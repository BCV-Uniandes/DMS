# -*- coding: utf-8 -*-

"""
Query-based Scene Segmentation (QSegNet) Network PyTorch implementation.
"""

import torch
import torch.nn as nn

from .vilstm import ViLSTM, ConvViLSTMCell
from .psp.pspnet import PSPNet, PSPUpsample


class QSegNet(nn.Module):
    def __init__(self, image_size, emb_size, hid_size, out_features=512,
                 num_vilstm_layers=2, pretrained=True, batch_first=True,
                 psp_size=1024, backend='densenet', dict_size=8054,
                 num_lstm_layers=2, dropout=0.2):
        super().__init__()
        self.psp = PSPNet(n_classes=1, psp_size=psp_size,
                          pretrained=pretrained, backend=backend,
                          out_features=out_features)
        self.emb = nn.Embedding(dict_size, emb_size)
        self.lstm = nn.LSTM(emb_size, hid_size, dropout=dropout,
                            batch_first=batch_first,
                            num_layers=num_lstm_layers)

        h, w = image_size
        self.vilstm = ViLSTM(ConvViLSTMCell, (h // 8, w // 8), 1, out_features,
                             vis_dim=out_features,
                             kernel_size=(3, 3),
                             num_layers=1,
                             batch_first=batch_first)

        self.up_1 = PSPUpsample(2 * out_features, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, 64)

        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1)
            # nn.LogSoftmax()
        )

    def forward(self, imgs, words):
        imgs = self.psp(imgs)

        words = self.emb(words)
        out, _ = self.lstm(words)

        # x_x is of size BxLxHxH
        # B: Batch Size
        # L: Phrase length
        # H: Hidden Size
        # x_x = torch.matmul(word_emb.unsqueeze(-1), word_emb.unsqueeze(2))
        # h_h = torch.matmul(out.unsqueeze(-1), out.unsqueeze(2))
        # h_x = torch.matmul(out.unsqueeze(-1), word_emb.unsqueeze(2))

        out = out.unsqueeze(-1).expand(out.size(0), out.size(1),
                                       out.size(2), out.size(2))
        # lang_input = h_h.unsqueeze(2)
        # lang_input = torch.cat(
        # [m.unsqueeze(2) for m in (x_x, h_h, h_x)], dim=2)
        # l_t: BxLx1024xHxH
        # l_t = self.lang_conv(lang_input)
        # mask: Bx1024xHxH
        _, (mask, _) = self.vilstm(out, imgs)

        if len(mask.size()) == 5:
            mask = mask.view(mask.size(0) * mask.size(1),
                             mask.size(2), mask.size(3),
                             mask.size(4))
        mask = torch.cat([imgs, mask], dim=1)
        p = self.up_1(mask)
        p = self.drop_2(p)

        p = self.up_2(p)
        p = self.drop_2(p)

        p = self.up_3(p)
        p = self.drop_2(p)

        return self.final(p)
        # return mask
