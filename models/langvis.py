# -*- coding: utf-8 -*-

"""
Query-based Scene Segmentation (QSegNet) Network PyTorch implementation.
"""

import torch
from sru import SRU
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .dpn.model_factory import create_model
import numpy as np


class LangVisNet(nn.Module):
    def __init__(self, dict_size, emb_size=1000, hid_size=1000,
                 vis_size=2688, num_filters=1, mixed_size=1000,
                 hid_mixed_size=1005, lang_layers=2, mixed_layers=3,
                 backend='dpn92', lstm=False, pretrained=True, 
                 extra=True, high_res=False, upsampling_channels=50):
        super().__init__()
        self.vis_size = vis_size
        self.num_filters = num_filters
        self.base = create_model(
            backend, 1, pretrained=pretrained, extra=extra)

        self.emb = nn.Embedding(dict_size, emb_size)
        self.lang_model = SRU(emb_size, hid_size, num_layers=lang_layers)
        if lstm:
            self.lang_model = nn.LSTM(
                emb_size, hid_size, num_layers=lang_layers)

        self.adaptative_filter = nn.Linear(
            in_features=hid_size, out_features=(num_filters * (vis_size + 2)))

        self.comb_conv = nn.Conv2d(in_channels=(2 + emb_size + hid_size +
                                                vis_size + num_filters),
                                   out_channels=mixed_size,
                                   kernel_size=1,
                                   padding=0)

        self.mrnn = SRU(mixed_size, hid_mixed_size,
                        num_layers=mixed_layers)
        if lstm:
            self.mrnn = nn.LSTM(mixed_size, hid_mixed_size,
                                num_layers=mixed_layers)
        self.output_collapse = nn.Conv2d(in_channels=hid_mixed_size,
                                         out_channels=1,
                                         kernel_size=1)

        if high_res:
            self.output_collapse = UpsamplingModule(in_channels=hid_mixed_size, 
                                                    upsampling_channels=upsampling_channels)

    def forward(self, vis, lang):
        B, C, H, W = vis.size()
        vis = self.base(vis)

        # LxE ?
        lang_mix = []
        lang = self.emb(lang)
        lang = torch.transpose(lang, 0, 1)
        lang_mix.append(lang.unsqueeze(-1).unsqueeze(-1).expand(
            lang.size(0), lang.size(1), lang.size(2),
            vis.size(-2), vis.size(-1)))
        # input has dimensions: seq_length x batch_size (1) x we_dim
        lang, _ = self.lang_model(lang)
        time_steps = lang.size(0)
        lang_mix.append(lang.unsqueeze(-1).unsqueeze(-1).expand(
            lang.size(0), lang.size(1), lang.size(2),
            vis.size(-2), vis.size(-1)))

        # Lx(H + E)xH/32xW/32
        lang_mix = torch.cat(lang_mix, dim=2)

        out_h, out_w = vis.size(2), vis.size(3)
        x = Variable(torch.linspace(start=-1, end=1, steps=out_w).cuda())
        x = x.unsqueeze(0).expand(out_h, out_w).unsqueeze(0).unsqueeze(0)

        y = Variable(torch.linspace(start=-1, end=1, steps=out_h).cuda())
        y = y.unsqueeze(1).expand(out_h, out_w).unsqueeze(0).unsqueeze(0)

        # (N + 2)xH/32xW/32
        vis = torch.cat([vis, x, y], dim=1)

        # Size: HxL?
        lang = lang.squeeze()
        filters = self.adaptative_filter(lang)
        filters = F.sigmoid(filters)
        # LxFx(N+2)x1x1
        filters = filters.view(
            time_steps, self.num_filters, self.vis_size + 2, 1, 1)
        p = []
        for t in range(time_steps):
            filter = filters[t]
            p.append(F.conv2d(input=vis, weight=filter).unsqueeze(0))

        # LxFxH/32xW/32
        p = torch.cat(p)

        # Lx(N + 2)xH/32xW/32
        vis = vis.unsqueeze(0).expand(time_steps, *vis.size())
        # Lx(N + F + H + E + 2)xH/32xW/32
        q = torch.cat([vis, lang_mix, p], dim=2)
        # Lx1xSxH/32xW/32
        # print(mixed.size())
        q = self.comb_conv(q.squeeze(1))
        q = q.unsqueeze(1)
        # q = []
        # for t in range(time_steps):
        #    q.append(self.comb_conv(mixed[t]).unsqueeze(0))
        # q = torch.cat(q)
        # LxSx((H + W)/32)
        # q = q.view(q.size(3) * q.size(4) * q.size(0), q.size(1), q.size(2))
        # Lx1xMxH/32xW/32
        q = q.view(q.size(0), q.size(1), q.size(2),
                   q.size(3) * q.size(4))
        # Lx1xMx(H*W/(32*32))
        q = q.permute(3, 0, 1, 2).contiguous()
        # q = torch.transpose(q, 3, 0)
        # q = torch.transpose(q, 3, 1)
        # q = torch.transpose(q, 3, 2).contiguous()
        # (H*W/(32*32))xLx1xM
        q = q.view(q.size(0) * q.size(1), q.size(2), q.size(3))
        # L*(H*W/(32*32))x1xM

        # input has dimensions: seq_length x batch_size x mix_size
        output, _ = self.mrnn(q)

        """
        Take all the hidden states (one for each pixel of every
        'length of the sequence') but keep only the last out_h * out_w
        so that it can be reshaped to an image of such size
        """
        output = output[-(out_h * out_w):, :, :]
        output = output.permute(1, 2, 0).contiguous()
        output = output.view(output.size(0), output.size(1),
                             out_h, out_w)

        output = self.output_collapse(output)
        return output

class UpsamplingModule(nn.Module):
    def __init__(self, in_channels, upsampling_channels, 
                 amplification=32, non_linearity=False):
        super().__init__()
        self.intermediate_modules = np.log2(amplification) - 2
        self.upsampling_channels = upsampling_channels
        self.non_linearity = non_linearity

        self.up = nn.Upsample(scale_factor=2, mode='nearest')

        self.first_conv = nn.Sequential(self.up,
                                        nn.Conv2d(in_channels=in_channels,
                                                  out_channels=upsampling_channels,
                                                  kernel_size=1))

        self.intermediate_convs = nn.ModuleList([
            self._make_conv() for _ in range(self.intermediate_modules)])
        
        self.final_conv = nn.Sequential(self.up,
                                        nn.Conv2d(in_channels=upsampling_channels,
                                                  out_channels=1,
                                                  kernel_size=1))

    def _make_conv(self):
        conv = nn.Conv2d(in_channels=self.upsampling_channels,
                         out_channels=self.upsampling_channels,
                         kernel_size=1)

        if self.non_linearity:
            conv = nn.Sequential(self.up, conv, nn.PReLU())
        else:
            conv = nn.Sequential(self.up, conv)

        return conv

    def forward(self, x):
        # Apply first convolution
        x = self.first_conv(x)

        # Apply intermediate convolutions
        for intermediate_conv in self.intermediate_convs:
            x = intermediate_conv(x)

        # Apply final convolution
        x = self.final_conv(x)      

        return x

