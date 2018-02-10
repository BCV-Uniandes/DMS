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
                 extra=True, gpu_pair=None, high_res=False):
        super().__init__()
        self.high_res = high_res
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

        if not self.high_res:
            self.output_collapse = nn.Conv2d(in_channels=hid_mixed_size,
                                             out_channels=1,
                                             kernel_size=1)

        self.gpu_pair = gpu_pair
        if gpu_pair is not None:
            # Define GPUs
            first_gpu = int(2*gpu_pair) # 0 if gpu_pair == 0 and 2 if gpu_pair == 1
            second_gpu = first_gpu + 1 # 1 if gpu_pair == 0 and 3 if gpu_pair == 1
            # Assign for use in forward
            self.first_gpu = first_gpu
            self.second_gpu = second_gpu
            # First GPU
            self.base.cuda(self.first_gpu)
            self.emb.cuda(self.first_gpu)
            self.lang_model.cuda(self.first_gpu)
            self.adaptative_filter.cuda(self.first_gpu)
            self.comb_conv.cuda(self.first_gpu)
            # Second GPU
            self.mrnn.cuda(self.second_gpu)
            if not self.high_res:
                self.output_collapse.cuda(self.second_gpu)

    def forward(self, vis, lang):
        # Run image through base FCN
        vis, base_features = self.base(vis)
        if self.gpu_pair is not None:
            vis = vis.cuda(self.first_gpu)

        # Generate channels of 'x' and 'y' info
        B, C, H, W = vis.size()
        spatial = self.generate_spatial_batch(H, W)
        if self.gpu_pair is not None:
            spatial = spatial.cuda(self.first_gpu)
        # (N + 8)xH/32xW/32
        vis = torch.cat([vis, spatial], dim=1)
        if self.gpu_pair is not None:
            vis = vis.cuda(self.first_gpu)

        # LxE ?
        linear_in = []
        lang_mix = []
        lang = self.emb(lang)
        if self.gpu_pair is not None:
            lang = lang.cuda(self.first_gpu)
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
        if self.gpu_pair is not None:
            lang_mix = lang_mix.cuda(self.first_gpu)

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
        if self.gpu_pair is not None:
            q = q.cuda(self.second_gpu)
            self.mrnn.cuda(self.second_gpu)
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
        if not self.high_res:
            if self.gpu_pair is not None:
                self.output_collapse.cuda(self.second_gpu)
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
            print(x.size(), features[i].size())
            # x = torch.cat([x, features[i]], dim=1)
            x = conv(x)
            i -= 1
        x = self.out_layer(x)
        return x


class LangVisUpsample(nn.Module):
    def __init__(self, dict_size, emb_size=1000, hid_size=1000,
                 vis_size=2688, num_filters=1, mixed_size=1000,
                 hid_mixed_size=1005, lang_layers=2, mixed_layers=3,
                 backend='dpn92', mix_we=False, lstm=False, pretrained=True,
                 extra=True, high_res=False, upsampling_channels=50,
                 upsampling_mode='bilineal', upsampling_size=3, gpu_pair=None,
                 upsampling_amplification=32, langvis_freeze=False):
        super().__init__()
        self.langvis = LangVisNet(dict_size, emb_size, hid_size,
                                  vis_size, num_filters, mixed_size,
                                  hid_mixed_size, lang_layers, mixed_layers,
                                  backend, mix_we, lstm, pretrained,
                                  extra, gpu_pair, high_res)
        self.high_res = high_res
        self.langvis_freeze = langvis_freeze
        if high_res:
            self.upsample = UpsamplingModule(
                hid_mixed_size, upsampling_channels, mode=upsampling_mode,
                ker_size=upsampling_size,
                amplification=upsampling_amplification)
        if langvis_freeze:
            self.langvis.eval()

    def forward(self, vis, lang):
        if self.langvis_freeze:
            vis = vis.detach()
            lang = lang.detach()
        out, features = self.langvis(vis, lang)
        if self.langvis_freeze:
            out = Variable(out.data)
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
