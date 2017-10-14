# -*- coding: utf-8 -*-

"""
Implementation of Visual Attention LSTM (ViLSTM)

Yuke Zhu, Oliver Groth, Michael S. Bernstein, Li Fei-Fei:
Visual7W: Grounded Question Answering in Images. CoRR abs/1511.03416 (2015)
https://arxiv.org/pdf/1511.03416.pd1f

Based on:
https://github.com/jihunchoi/recurrent-batch-normalization-pytorch/blob/master/bnlstm.py
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


class LangConv(nn.Module):
    def __init__(self, out_size):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.pool = nn.AdaptiveMaxPool2d(output_size=(out_size, out_size))
        # self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2,
        #                        padding=3, bias=False)
        # self.conv3 = nn.Conv2d(128, 256, kernel_size=7, stride=1, padding=3,
        #                        bias=False)
        # self.conv4 = nn.Conv2d(256, 512, kernel_size=7, stride=1, padding=3,
        #                        bias=False)
        # self.conv5 = nn.Conv2d(512, 1024, kernel_size=7, stride=1, padding=1,
        #                        bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # x = self.conv5(x)
        x = self.pool(x)
        return x


class ConvViLSTMCell(nn.Module):
    """Basic Convolutional Visual Attention LSTM cell."""

    def __init__(self, input_size, input_dim, hidden_dim, vis_dim,
                 kernel_size, bias):
        """
        Initialize ConvViLSTM cell.

        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        vis_dim: int
            Number of channels of visual tensor.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvViLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.vis_dim = vis_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        # self.lang_conv = LangConv(input_size)
        self.e_conv = nn.Conv2d(in_channels=self.vis_dim + self.hidden_dim,
                                out_channels=self.hidden_dim,
                                kernel_size=self.kernel_size,
                                padding=self.padding,
                                bias=False)
        self.a_conv = nn.Conv2d(in_channels=self.hidden_dim,
                                out_channels=self.hidden_dim,
                                kernel_size=self.kernel_size,
                                padding=self.padding,
                                bias=self.bias)
        self.conv = nn.Conv2d(in_channels=self.input_dim + 2 * self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_, features, hx):
        h_cur, c_cur = hx

        input_ = input_.squeeze().unsqueeze(1)
        lang_hid = torch.cat([h_cur, features], dim=1)
        v = self.e_conv(lang_hid)
        v = self.a_conv(torch.tanh(v))
        v = F.softmax(v)
        v = v * features
        # concatenate along channel axis
        combined = torch.cat([input_, h_cur, v], dim=1)
        combined = self.conv(combined)
        i, f, o, g = torch.split(
            combined, self.hidden_dim, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_cur = f * c_cur + i * g
        h_cur = o * torch.tanh(c_cur)

        return h_cur, c_cur


class ViLSTM(nn.Module):

    """A module that runs multiple steps of ViLSTM."""

    def __init__(self, cell_class, input_size, input_dim, hidden_dim,
                 num_layers=1, use_bias=True, batch_first=False,
                 dropout=0, **kwargs):
        super(ViLSTM, self).__init__()
        self.input_size = input_size
        self.cell_class = cell_class
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_bias = use_bias
        self.batch_first = batch_first
        self.dropout = dropout

        for layer in range(num_layers):
            layer_input_size = input_dim if layer == 0 else hidden_dim
            cell = cell_class(input_size=self.input_size,
                              input_dim=layer_input_size,
                              hidden_dim=hidden_dim,
                              bias=self.use_bias,
                              **kwargs)
            setattr(self, 'cell_{}'.format(layer), cell)
        self.dropout_layer = nn.Dropout(dropout)
        # self.reset_parameters()

    def get_cell(self, layer):
        return getattr(self, 'cell_{}'.format(layer))

    def reset_parameters(self):
        for layer in range(self.num_layers):
            cell = self.get_cell(layer)
            cell.reset_parameters()

    @staticmethod
    def _forward_rnn(cell, input_, features, length, hx):
        max_time = input_.size(0)
        output = []
        for time in range(max_time):
            print(time)
            # if isinstance(cell, BNLSTMCell):
            #     h_next, c_next = cell(input_=input_[time], hx=hx, time=time)
            # else:
            h_next, c_next = cell(input_[time], features, hx)
            mask = (time < length).float()
            mask = mask.view(-1, 1, 1, 1)
            mask = mask.expand_as(h_next)
            h_next = h_next * mask + hx[0] * (1 - mask)
            c_next = c_next * mask + hx[1] * (1 - mask)
            hx = (h_next, c_next)
            output.append(hx)
        output = torch.stack(output, 0)
        return output, hx

    def forward(self, input_, features, length=None, hx=None):
        if self.batch_first:
            input_ = input_.transpose(0, 1)
        max_time, batch_size = input_.size()[:2]
        if length is None:
            length = Variable(torch.LongTensor([max_time] * batch_size))
            if input_.is_cuda:
                device = input_.get_device()
                length = length.cuda(device)
        if hx is None:
            hx = Variable(input_.data.new(
                batch_size, self.hidden_dim, *self.input_size).zero_())
            hx = (hx, hx)
        h_n = []
        c_n = []
        layer_output = None
        for layer in range(self.num_layers):
            cell = self.get_cell(layer)
            layer_output, (layer_h_n, layer_c_n) = ViLSTM._forward_rnn(
                cell=cell, input_=input_, features=features,
                length=length, hx=hx)
            input_ = self.dropout_layer(layer_output)
            h_n.append(layer_h_n)
            c_n.append(layer_c_n)
        output = layer_output
        h_n = torch.stack(h_n, 0)
        c_n = torch.stack(c_n, 0)
        return output, (h_n, c_n)
