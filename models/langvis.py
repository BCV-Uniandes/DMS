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
    def __init__(self):
        super().__init__()
        if backend == 'dpn92':
            self.base = create_model(
                backend, 1, pretrained=pretrained, extra=extra)
        else:
            self.base = create_model(
                backend, 1, pretrained=pretrained)

        self.output_collapse = nn.Conv2d(in_channels=2688,
                                             out_channels=1,
                                             kernel_size=1)

    def forward(self, vis, lang):
        # Run image through base FCN
        vis, base_features = self.base(vis)
        if self.gpu_pair is not None:
            vis = vis.cuda(self.first_gpu)

        # Generate channels of 'x' and 'y' info
        B, C, H, W = vis.size()
        spatial = self.generate_spatial_batch(H, W)
        
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
        spatial_batch_val = np.zeros((1, 8, featmap_H, featmap_W), dtype=np.float32)
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
        return Variable(torch.from_numpy(spatial_batch_val)).cuda()


class LangVisUpsample(nn.Module):
    def __init__(self):
        super().__init__()
        self.langvis = LangVisNet()

    def forward(self, vis, lang):
        out, _ = self.langvis(vis, lang)
        return out

    def load_state_dict(self, new_state):
        state = self.state_dict()
        for layer in state:
            if layer in new_state:
                if state[layer].size() == new_state[layer].size():
                    state[layer] = new_state[layer]
        super().load_state_dict(state)
