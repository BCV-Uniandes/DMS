# -*- coding: utf-8 -*-

"""Dynamic Multimodal Network (DMN) PyTorch implementation."""

import torch
import numpy as np
from sru import SRU
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .dpn.model_factory import create_model


class BaseDMN(nn.Module):
    r"""
    Base DMN implementation: Language, Visual and Multimodal modules.

    Given an image :math:`I` and a referral expression :math:`e` encoded as a
    number sequence of length :math:`T`, the Base DMN generates a low
    resolution segmentation mask :math:`M_5` for the object referred on the
    input expression.

    .. math::
        \begin{array}{ll}
        (I_1, I_2, I_3, I_4, I_5) = V(I) \\
        h_t = RNN(WE(e_t), h_{t - 1}) \quad t = 0, \cdots, T \\
        r_t = \left[ h_t, WE(e_t) \right]
        f_t = \left\{\sigma(W_{k} r_t + b_{k})\right\}_{k = 1} ^ K \\
        F_t = I_5 * f_t
        M_t = \text{Conv}_{1 \times 1}\left([I_5, F_t, LOC, r_t]\right) \\
        R_5 = mRNN(\left\{M_t\right\}_{t = 1}^{T})
        \end{array}

    where :math:`V` is the visual module, :math:`I_j` are the output responses
    per each downsampled resolution level (2x, ..., 32x), :math:`*` is the
    convolution operator and :math:`\sigma` is the logistic sigmoid function.

    Args:
        dict_size: The total number of words of the input expression
            dictionary encoding
        emb_size: The size of each word embedding vector. Default: 1000
        hid_size: The number of features in the hidden state `h`. Default: 1000
        vis_size: The number of output channels of the visual CNN module.
            Default: 2688
        num_filters: The number of dynamical filters to compute based on the
            language features. Default: 10
        mixed_size: The number of input features to multimodal RNN.
            Default: 1000
        hid_mixed_size: The number of features in the multimodal hidden state.
            Default: 1005
        lang_layers: Number of recurrent layers in the Language RNN. Default: 3
        mixed_layers: Number of recurrent layers in the Multimodal RNN.
            Default: 3
        backend: Name of the visual module backbone to use.
            See :class:`models.dpn.model_factory` to see all the available
            visual backbones. Default: `dpn92`
        mix_we: If ``True``, then the word embeddings are used alongside the
            language hidden states to compute each dynamical filter.
            Default: ``True``
        lstm: If ``True``, then LSTM units are used for both language and
            multimodal RNN modules. Otherwise it uses SRU units.
            Default: ``False``
        pretrained: If ``False``, the Visual Module weights are initialized
            randomly and not from ImageNet weights. Default: ``True``
        extra: Reserved parameter to initialize dpn92 modules.
            Default: ``True``
        high_res: If ``True``, then the last multimodal hidden state is
            returned as-is. Otherwise, it will apply an additional convolution
            to produce a low-resolution segmentation map with a single channel.
            Default: ``False``

    Inputs: vis, lang
        - **vis** of shape :math:`(1, 3, H, W)`: tensor containing an input
        image in format RGB.
        - **lang** of shape :math:`(1, L)`: tensor containing an referral
        expression given as a number sequence.

    Outputs: output, base_features
        - **output** of shape :math:`(O, H/32, W/32)`: tensor containing
        :math:`O` low resolution segmentation map features. If `high_res` is
        ``False``, then :math:`O = 1`.
        - **output**: a list containing all the downsampled feature maps
        returned by the visual module as tensors of shape
        :math:`(1, C, H/2^k, W/2^k)`, with :math:`k = 1, \cdots, 5`.
    """

    def __init__(self, dict_size, emb_size=1000, hid_size=1000,
                 vis_size=2688, num_filters=10, mixed_size=1000,
                 hid_mixed_size=1005, lang_layers=3, mixed_layers=3,
                 backend='dpn92', mix_we=True, lstm=False, pretrained=True,
                 extra=True, high_res=False):
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
        self.lang_model = SRU(emb_size, hid_size, num_layers=lang_layers,
                              rescale=False)
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

    def forward(self, vis, lang):
        # Run image through base FCN
        # vis: BxCxHxW
        vis, base_features = self.base(vis)

        # Generate channels of 'x' and 'y' info
        _, _, H, W = vis.size()
        spatial = self.generate_spatial_batch(H, W)

        # Add additional visual hint feature maps.
        # vis: (N + 8)xH/32xW/32
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
        # Size: HxL?
        linear_in = linear_in.squeeze()
        filters = self.adaptative_filter(linear_in)
        filters = F.sigmoid(filters)
        # LxFx(N+2)x1x1
        filters = filters.view(
            time_steps, self.num_filters, self.vis_size + 8, 1, 1)
        p = []
        for t in range(time_steps):
            _filter = filters[t]
            p.append(F.conv2d(input=vis, weight=_filter).unsqueeze(0))

        # LxFxH/32xW/32
        p = torch.cat(p)

        # Lx(N + 2)xH/32xW/32
        vis = vis.unsqueeze(0).expand(time_steps, *vis.size())
        # Lx(N + F + H + E + 2)xH/32xW/32
        q = torch.cat([vis, lang_mix, p], dim=2)
        # Lx1xSxH/32xW/32

        q = self.comb_conv(q.squeeze(1))
        q = q.unsqueeze(1)

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

        # Take all the hidden states (one for each pixel of every
        # 'length of the sequence') but keep only the last H * W
        # so that it can be reshaped to an image of such size
        output = output[-(H * W):, :, :]
        output = output.permute(1, 2, 0).contiguous()
        output = output.view(output.size(0), output.size(1),
                             H, W)
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

    @classmethod
    def generate_spatial_batch(self, featmap_H, featmap_W):
        """Generate additional visual coordinates feature maps.

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
        return Variable(torch.from_numpy(spatial_batch_val)).cuda()


class UpsamplingModule(nn.Module):
    r"""
    DMN Upsampling Module.

    Given a low resolution segmentation mask :math:`M_5` and a set
    of pyramid features :math:`(I_k)_{i = 1}^{5}`, the Upsampling Module
    computes a high resolution segmentation mask :math:`M_1` on a incremental
    fashion. At each step, the module takes the current resolution map and
    combines it with its corresponding feature map.

    .. math::
        \begin{array}{ll}
        M_k = \text{Upsample}(\text{Conv}[M_{k+1}, I_{k+1}])) \quad
        k = 4, \cdots, 1
        \end{array}

    Args:
        in_channels: The total number of input features incoming from the
            multimodal module
        mode: Usampling mode, see :class:`torch.nn.Upsample`.
            Default: `bilineal`
        ker_size: Kernel size for each convolution step. Default: 3
        amplification: Amplification zoom to apply, it must be a power
            of two. Default: 32
        non_linearity: If ``True``, it will apply a :class:`torch.nn.PReLU`
            activation in-between upsampling steps. Default: ``False``
        feature_channels: Expected number of feature channels per
            each visual map. Default:  `[2688, 1552, 704, 336, 64]`

    Inputs: x, features
        - **x** of shape :math:`(O, H/32, W/32)`: tensor containing
        :math:`O` low resolution segmentation map features.
        - **features** a list containing all the downsampled feature maps
        returned by the visual module as tensors of shape
        :math:`(1, C, H/2^k, W/2^k)`, with :math:`k = 1, \cdots, 5`.

    Outputs: mask
        - **output** of shape :math:`(1, H/\log_2{ampl}, W/\log_2{ampl})`:
        tensor that contains a single segmentation mask map.
    """

    def __init__(self, in_channels, mode='bilineal', ker_size=3,
                 amplification=32, non_linearity=False,
                 feature_channels=[2688, 1552, 704, 336, 64]):
        super().__init__()
        self.ker_size = ker_size
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
                    mode='bilinear')
            x = torch.cat([x, features[i]], dim=1)
            x = conv(x)
            i -= 1
        x = self.out_layer(x)
        return x


class DMN(nn.Module):
    r"""
    Dynamic Multimodal Network (DMN).

    Given an image :math:`I` and a referral expression :math:`e` encoded as a
    number sequence of length :math:`T`, the Base DMN generates a low
    resolution segmentation mask :math:`M_1.

    .. math::
        \begin{array}{ll}
        (I_1, I_2, I_3, I_4, I_5) = V(I) \\
        h_t = RNN(WE(e_t), h_{t - 1}) \quad t = 0, \cdots, T \\
        r_t = \left[ h_t, WE(e_t) \right]
        f_t = \left\{\sigma(W_{k} r_t + b_{k})\right\}_{k = 1} ^ K \\
        F_t = I_5 * f_t
        M_t = \text{Conv}_{1 \times 1}\left([I_5, F_t, LOC, r_t]\right) \\
        R_5 = mRNN(\left\{M_t\right\}_{t = 1}^{T})
        M_j = \text{Upsample}(\text{Conv}[M_{j+1}, I_{j+1}])) \quad
        j = 4, \cdots, 1
        \end{array}

    where :math:`V` is the visual module, :math:`I_j` are the output responses
    per each downsampled resolution level (2x, ..., 32x), :math:`*` is the
    convolution operator and :math:`\sigma` is the logistic sigmoid function.

    Args:
        dict_size: The total number of words of the input expression
            dictionary encoding
        emb_size: The size of each word embedding vector. Default: 1000
        hid_size: The number of features in the hidden state `h`. Default: 1000
        vis_size: The number of output channels of the visual CNN module.
            Default: 2688
        num_filters: The number of dynamical filters to compute based on the
            language features. Default: 10
        mixed_size: The number of input features to multimodal RNN.
            Default: 1000
        hid_mixed_size: The number of features in the multimodal hidden state.
            Default: 1005
        lang_layers: Number of recurrent layers in the Language RNN. Default: 3
        mixed_layers: Number of recurrent layers in the Multimodal RNN.
            Default: 3
        backend: Name of the visual module backbone to use.
            See :class:`models.dpn.model_factory` to see all the available
            visual backbones. Default: `dpn92`
        mix_we: If ``True``, then the word embeddings are used alongside the
            language hidden states to compute each dynamical filter.
            Default: ``True``
        lstm: If ``True``, then LSTM units are used for both language and
            multimodal RNN modules. Otherwise it uses SRU units.
            Default: ``False``
        pretrained: If ``False``, the Visual Module weights are initialized
            randomly and not from ImageNet weights. Default: ``True``
        extra: Reserved parameter to initialize dpn92 modules.
            Default: ``True``
        high_res: If ``True``, then the last multimodal hidden state is
            returned as-is. Otherwise, it will apply an additional convolution
            to produce a low-resolution segmentation map with a single channel.
            Default: ``False``
        upsampling_mode: Usampling mode, see :class:`torch.nn.Upsample`.
            Default: `bilineal`
        upsampling_size: Kernel size for each upsampling convolution step.
            Default: 3
        upsampling_amplification: Amplification zoom to apply, it must be a
            power of two. Default: 32
        dmn_freeze: If ``True``, only the upsampling module is trained.
            Default: ``False``


    Inputs: vis, lang
        - **vis** of shape :math:`(1, 3, H, W)`: tensor containing an input
        image in format RGB.
        - **lang** of shape :math:`(1, L)`: tensor containing an referral
        expression given as a number sequence.

    Outputs: out
        - **output** of shape :math:`(1, H/\log_2{ampl}, W/\log_2{ampl})`:
        tensor that contains a single segmentation mask map.
    """

    def __init__(self, dict_size, emb_size=1000, hid_size=1000,
                 vis_size=2688, num_filters=1, mixed_size=1000,
                 hid_mixed_size=1005, lang_layers=2, mixed_layers=3,
                 backend='dpn92', mix_we=False, lstm=False, pretrained=True,
                 extra=True, high_res=False,
                 upsampling_mode='bilineal', upsampling_size=3,
                 upsampling_amplification=32, dmn_freeze=False):
        super().__init__()
        self.langvis = BaseDMN(dict_size, emb_size, hid_size,
                               vis_size, num_filters, mixed_size,
                               hid_mixed_size, lang_layers, mixed_layers,
                               backend, mix_we, lstm, pretrained,
                               extra, high_res)
        self.high_res = high_res
        self.dmn_freeze = dmn_freeze
        if high_res:
            self.upsample = UpsamplingModule(
                hid_mixed_size, mode=upsampling_mode,
                ker_size=upsampling_size,
                amplification=upsampling_amplification)
        if dmn_freeze:
            self.langvis.eval()

    def forward(self, vis, lang):
        if self.dmn_freeze:
            vis = vis.detach()
            lang = lang.detach()
        out, features = self.langvis(vis, lang)
        if self.dmn_freeze:
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
