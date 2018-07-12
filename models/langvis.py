# -*- coding: utf-8 -*-

"""
Language and Vision (LangVisNet) Network PyTorch implementation.
"""

import torch
import numpy as np
from sru import SRU
import torch.nn as nn
import torch.nn.functional as F
from .dpn.model_factory import create_model


class LangVisNet(nn.Module):
    def __init__(self, backend='dpn92', pretrained=True, extra=True,
                 high_res=False, vis_size=2688):
        super().__init__()
        self.high_res = high_res
        self.vis_size = vis_size
        self.visual_freeze = False
        if backend == 'dpn92':
            self.base = create_model(
                backend, 1, pretrained=pretrained, extra=extra)
        else:
            self.base = create_model(
                backend, 1, pretrained=pretrained)

        if not self.high_res:
            self.output_collapse = nn.Conv2d(in_channels=self.vis_size,
                                             out_channels=1,
                                             kernel_size=1)

    def forward(self, vis, lang):
        # Run image through base FCN
        # with torch.set_grad_enabled(self.visual_freeze):
        output, base_features = self.base(vis)
        output = output.requires_grad_()

        # Generate channels of 'x' and 'y' info
        B, C, H, W = output.size()
        # spatial = self.generate_spatial_batch(H, W)
        if not self.high_res:
            output = self.output_collapse(output)
        return output, base_features

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
        spatial_batch_val = np.zeros(
            (1, 8, featmap_H, featmap_W), dtype=np.float32)
        for h in range(featmap_H):
            for w in range(featmap_W):
                xmin = w / featmap_W * 2 - 1
                xmax = (w + 1) / featmap_W * 2 - 1
                xctr = (xmin + xmax) / 2
                ymin = h / featmap_H * 2 - 1
                ymax = (h + 1) / featmap_H * 2 - 1
                yctr = (ymin + ymax) / 2
                spatial_batch_val[0, :, h, w] = (
                    [xmin, ymin, xmax, ymax,
                     xctr, yctr, 1 / featmap_W, 1 / featmap_H])
        return torch.from_numpy(spatial_batch_val).cuda()


class UpsamplingModule(nn.Module):
    def __init__(self, in_channels, upsampling_channels=1,
                 mode='bilineal', ker_size=3,
                 amplification=32, non_linearity=False,
                 feature_channels=[2688, 1552, 704, 336, 64]):
        super().__init__()
        self.ker_size = ker_size
        self.upsampling_channels = upsampling_channels
        self.non_linearity = non_linearity
        self.up = nn.Upsample(scale_factor=2, mode=mode)
        self.convs = []
        num_layers = int(np.log2(amplification))

        i = 0
        for out_channels in np.logspace(
                9, 10 - num_layers, num=num_layers, base=2, dtype=int):
            self.convs.append(self._make_conv(
                int(in_channels) + feature_channels[i], int(out_channels)))
            i += 1
            in_channels = int(out_channels)

        self.out_layer = nn.Conv2d(in_channels=in_channels,
                                   out_channels=1,
                                   kernel_size=1,
                                   padding=0)
        # self.convs.append(out_layer)
        self.convs = nn.ModuleList(self.convs)

    def _make_conv(self, in_channels, out_channels):
        conv = nn.Conv2d(in_channels=in_channels,
                         out_channels=out_channels,
                         kernel_size=self.ker_size,
                         padding=(self.ker_size // 2))

        if self.non_linearity:
            conv = nn.Sequential(self.up, conv, nn.PReLU())
        else:
            conv = nn.Sequential(self.up, conv)

        return conv

    def forward(self, x, features):
        # Apply all layers
        i = len(features) - 1
        for conv in self.convs:
            if ((x.size(-2), x.size(-1)) != (
                    features[i].size(-2), features[i].size(-1))):
                x = F.upsample(
                    x, (features[i].size(-2), features[i].size(-1)),
                    mode='bilinear', align_corners=True)
            x = torch.cat([x, features[i]], dim=1)
            x = conv(x)
            i -= 1
        x = self.out_layer(x)
        return x


class LangVisUpsample(nn.Module):
    def __init__(self, backend='dpn92', vis_size=2688, high_res=False):
        super().__init__()
        self.high_res = high_res
        self.langvis = LangVisNet(high_res=high_res, backend=backend,
                                  vis_size=vis_size)
        if self.high_res:
            self.upsample = UpsamplingModule(vis_size, 50,
                                             'bilinear', 3, 32,
                                             False)
    def forward(self, vis, lang):
        out, features = self.langvis(vis, lang)
        if self.high_res:
            out = self.upsample(out, features)
        return out

    def load_state_dict(self, new_state):
        state = self.state_dict()
        for layer in state:
            if layer in new_state:
                if state[layer].size() == new_state[layer].size():
                    state[layer] = new_state[layer]
        super().load_state_dict(state)
