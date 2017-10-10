# -*- coding: utf-8 -*-

"""
Implementation of Visual Attention LSTM (VILSTM)

Yuke Zhu, Oliver Groth, Michael S. Bernstein, Li Fei-Fei:
Visual7W: Grounded Question Answering in Images. CoRR abs/1511.03416 (2015)
https://arxiv.org/pdf/1511.03416.pd1f

Based on:
https://github.com/jihunchoi/recurrent-batch-normalization-pytorch/blob/master/bnlstm.py
"""

import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


class VILSTMCell(nn.Module):

    """A basic Visual Attention LSTM cell."""

    def __init__(self, input_size, hidden_size, visual_size,
                 mix_size, use_bias=True):
        """
        Most parts are copied from torch.nn.LSTMCell.
        """

        super(VILSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        self.weight_a = nn.Parameter(
            torch.FloatTensor(mix_size, visual_size))
        self.weight_he = nn.Parameter(
            torch.FloatTensor(hidden_size, mix_size))
        self.weight_ce = nn.Parameter(
            torch.FloatTensor(visual_size, mix_size))
        self.weight_ih = nn.Parameter(
            torch.FloatTensor(input_size, 4 * hidden_size))
        self.weight_hh = nn.Parameter(
            torch.FloatTensor(hidden_size, 4 * hidden_size))
        self.weight_vh = nn.Parameter(
            torch.FloatTensor(visual_size, 4 * hidden_size))
        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(4 * hidden_size))
            self.comp_bias = nn.Parameter(torch.FloatTensor(mix_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input_, features, hx):
        """
        Args:
            input_: A (batch, input_size) tensor containing input
                features.
            features: A (batch, visual_size) tensor containing convolutional
                features.
            hx: A tuple (h_0, c_0), which contains the initial hidden
                and cell state, where the size of both states is
                (batch, hidden_size).

        Returns:
            h_1, c_1: Tensors containing the next hidden and cell state.
        """

        h_0, c_0 = hx
        batch_size = h_0.size(0)
        bias_batch = (self.bias.unsqueeze(0)
                      .expand(batch_size, *self.bias.size()))
        wh_b = torch.addmm(bias_batch, h_0, self.weight_hh)
        wi = torch.mm(input_, self.weight_ih)

        h_e = torch.mm(h_0, self.weight_he)
        c_e = torch.mm(features, self.weight_ce)
        e_i = torch.add(torch.tanh(h_e + c_e), self.comp_bias)
        e = torch.mm(e_i, self.weight_a)
        a = F.softmax(e)
        v_0 = features * a

        wv = torch.mm(v_0, self.weight_vh)
        f, i, o, g = torch.split(wh_b + wi + wv,
                                 split_size=self.hidden_size, dim=1)
        c_1 = torch.sigmoid(f) * c_0 + torch.sigmoid(i) * torch.tanh(g)
        h_1 = torch.sigmoid(o) * torch.tanh(c_1)
        return h_1, c_1

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class VILSTM(nn.Module):

    """A module that runs multiple steps of VILSTM."""

    def __init__(self, cell_class, input_size, hidden_size, num_layers=1,
                 use_bias=True, batch_first=False, dropout=0, **kwargs):
        super(VILSTM, self).__init__()
        self.cell_class = cell_class
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_bias = use_bias
        self.batch_first = batch_first
        self.dropout = dropout

        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            cell = cell_class(input_size=layer_input_size,
                              hidden_size=hidden_size,
                              **kwargs)
            setattr(self, 'cell_{}'.format(layer), cell)
        self.dropout_layer = nn.Dropout(dropout)
        self.reset_parameters()

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
            # if isinstance(cell, BNLSTMCell):
            #     h_next, c_next = cell(input_=input_[time], hx=hx, time=time)
            # else:
            h_next, c_next = cell(
                input_=input_[time], features=features, hx=hx)
            mask = (time < length).float().unsqueeze(1).expand_as(h_next)
            h_next = h_next * mask + hx[0] * (1 - mask)
            c_next = c_next * mask + hx[1] * (1 - mask)
            hx_next = (h_next, c_next)
            output.append(h_next)
            hx = hx_next
        output = torch.stack(output, 0)
        return output, hx

    def forward(self, input_, features, length=None, hx=None):
        if self.batch_first:
            input_ = input_.transpose(0, 1)
        max_time, batch_size, _ = input_.size()
        if length is None:
            length = Variable(torch.LongTensor([max_time] * batch_size))
            if input_.is_cuda:
                device = input_.get_device()
                length = length.cuda(device)
        if hx is None:
            hx = Variable(input_.data.new(
                batch_size, self.hidden_size).zero_())
            hx = (hx, hx)
        h_n = []
        c_n = []
        layer_output = None
        for layer in range(self.num_layers):
            cell = self.get_cell(layer)
            layer_output, (layer_h_n, layer_c_n) = VILSTM._forward_rnn(
                cell=cell, input_=input_, features=features,
                length=length, hx=hx)
            input_ = self.dropout_layer(layer_output)
            h_n.append(layer_h_n)
            c_n.append(layer_c_n)
        output = layer_output
        h_n = torch.stack(h_n, 0)
        c_n = torch.stack(c_n, 0)
        return output, (h_n, c_n)
