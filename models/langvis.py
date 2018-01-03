# -*- coding: utf-8 -*-

"""
Language and Vision (LangVisNet) Network PyTorch implementation.
"""

import torch
import numpy as np
from sru import SRU
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .dpn.model_factory import create_model


class LangVisNet(nn.Module):
    def __init__(self, dict_size, emb_size=1000, hid_size=1000,
                 vis_size=2688, num_filters=1, mixed_size=1000,
                 hid_mixed_size=1005, lang_layers=2, mixed_layers=3,
                 backend='dpn92', mix_we=False, lstm=False, pretrained=True,
                 extra=True, high_res=False, upsampling_channels=50,
                 upsampling_mode='bilineal', upsampling_size=3):
        super().__init__()
        self.vis_size = vis_size
        self.num_filters = num_filters
        if backend == 'dpn92':
            self.base = create_model(
                backend, 1, pretrained=pretrained, extra=extra)
        else:
            self.base = create_model(
                backend, 1, pretrained=pretrained)

        self.emb = nn.Embedding(dict_size, emb_size)
        self.lang_model = SRU(emb_size, hid_size, num_layers=lang_layers)
        if lstm:
            self.lang_model = nn.LSTM(
                emb_size, hid_size, num_layers=lang_layers)

        self.mix_we = mix_we
        lineal_in = hid_size + emb_size * int(mix_we)
        self.adaptative_filter = nn.Linear(
            in_features=lineal_in, out_features=(num_filters * (vis_size + 8)))

        self.comb_conv = nn.Conv2d(in_channels=(8 + emb_size + hid_size +
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
            self.output_collapse = UpsamplingModule(
                in_channels=hid_mixed_size,
                upsampling_channels=upsampling_channels,
                mode=upsampling_mode,
                ker_size=upsampling_size)

    def forward(self, vis, lang):
        # Run image through base FCN
        vis = self.base(vis)

        # Generate channels of 'x' and 'y' info
        B, C, H, W = vis.size()
        spatial = self.generate_spatial_batch(H, W)
        # (N + 8)xH/32xW/32
        vis = torch.cat([vis, spatial], dim=1)

        # LxE ?
        linear_in = []
        lang_mix = []
        lang = self.emb(lang)
        lang = torch.transpose(lang, 0, 1)
        if self.mix_we:
            linear_in.append(lang.squeeze(dim=1))
        lang_mix.append(lang.unsqueeze(-1).unsqueeze(-1).expand(
            lang.size(0), lang.size(1), lang.size(2),
            vis.size(-2), vis.size(-1)))
        # input has dimensions: seq_length x batch_size (1) x we_dim
        lang, _ = self.lang_model(lang)
        # Lx1xH
        time_steps = lang.size(0)
        lang_mix.append(lang.unsqueeze(-1).unsqueeze(-1).expand(
            lang.size(0), lang.size(1), lang.size(2),
            vis.size(-2), vis.size(-1)))

        if self.mix_we:
            linear_in.append(lang.squeeze(dim=1))
            linear_in = torch.cat(linear_in, dim=1)
        else:
            linear_in = lang

        # Lx(H + E)xH/32xW/32
        lang_mix = torch.cat(lang_mix, dim=2)

        # x = Variable(torch.linspace(start=-1, end=1, steps=out_w).cuda())
        # x = x.unsqueeze(0).expand(out_h, out_w).unsqueeze(0).unsqueeze(0)

        # y = Variable(torch.linspace(start=-1, end=1, steps=out_h).cuda())
        # y = y.unsqueeze(1).expand(out_h, out_w).unsqueeze(0).unsqueeze(0)
        

        # Size: HxL?
        linear_in = linear_in.squeeze()
        # if self.mix_we:
        filters = self.adaptative_filter(linear_in)
        # else:
            # filters = self.adaptative_filter(lang)
        filters = F.sigmoid(filters)
        # LxFx(N+2)x1x1
        filters = filters.view(
            time_steps, self.num_filters, self.vis_size + 8, 1, 1)
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
        # (H*W/(32*32))xLx1xM
        q = q.view(q.size(0) * q.size(1), q.size(2), q.size(3))
        # L*(H*W/(32*32))x1xM

        # input has dimensions: seq_length x batch_size x mix_size
        output, _ = self.mrnn(q)

        """
        Take all the hidden states (one for each pixel of every
        'length of the sequence') but keep only the last H * W
        so that it can be reshaped to an image of such size
        """
        output = output[-(H * W):, :, :]
        output = output.permute(1, 2, 0).contiguous()
        output = output.view(output.size(0), output.size(1),
                             H, W)

        output = self.output_collapse(output)
        return output

    def load_state_dict(self, new_state):
        state = self.state_dict()
        for layer in state:
            if layer in new_state:
                if state[layer].size() == new_state[layer].size():
                    state[layer] = new_state[layer]
        super().load_state_dict(state)

    def generate_spatial_batch(self, featmap_H, featmap_W):
        """
        Function taken from 
        https://github.com/chenxi116/TF-phrasecut-public/blob/master/util/processing_tools.py#L5
        and slightly modified
        """
        spatial_batch_val = np.zeros((1, 8, featmap_H, featmap_W), dtype=np.float32)
        for h in range(featmap_H):
            for w in range(featmap_W):
                xmin = w / featmap_W * 2 - 1
                xmax = (w+1) / featmap_W * 2 - 1
                xctr = (xmin+xmax) / 2
                ymin = h / featmap_H * 2 - 1
                ymax = (h+1) / featmap_H * 2 - 1
                yctr = (ymin+ymax) / 2
                spatial_batch_val[0, :, h, w] = \
                    [xmin, ymin, xmax, ymax, xctr, yctr, 1/featmap_W, 1/featmap_H]
        return Variable(torch.from_numpy(spatial_batch_val)).cuda()


class UpsamplingModule(nn.Module):
    def __init__(self, in_channels, upsampling_channels,
                 mode='bilineal', ker_size=3,
                 amplification=32, non_linearity=False):
        super().__init__()
        self.ker_size = ker_size
        self.intermediate_modules = int(np.log2(amplification) - 2)
        self.upsampling_channels = upsampling_channels
        self.non_linearity = non_linearity

        self.up = nn.Upsample(scale_factor=2, mode=mode)

        self.first_conv = nn.Sequential(self.up,
                                        nn.Conv2d(
                                            in_channels=in_channels,
                                            out_channels=upsampling_channels,
                                            kernel_size=self.ker_size))

        self.intermediate_convs = nn.ModuleList([
            self._make_conv() for _ in range(self.intermediate_modules)])

        self.final_conv = nn.Sequential(self.up,
                                        nn.Conv2d(
                                            in_channels=upsampling_channels,
                                            out_channels=1,
                                            kernel_size=self.ker_size))

    def _make_conv(self):
        conv = nn.Conv2d(in_channels=self.upsampling_channels,
                         out_channels=self.upsampling_channels,
                         kernel_size=self.ker_size)

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
