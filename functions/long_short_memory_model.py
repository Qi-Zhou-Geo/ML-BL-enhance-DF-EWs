#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2024-02-23
#__author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
#__find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd
# Please do not distribute this code without the author's permission

import torch
import torch.nn as nn
from torch.nn import functional as F


class RNN_LSTM(nn.Module):
    def __init__(self, feature_size, dropout, hidden_num=256, layer_num=2, output_dim=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=feature_size,
            hidden_size=hidden_num,
            layer_num=layer_num,
            bias=True,
            batch_first=True,
            dropout=dropout,
            bidirectional=False).to(torch.float64)
        self.layer_num = layer_num
        self.hidden_size = hidden_num
        self.fully_connect = nn.Linear(hidden_num, output_dim).to(torch.float64)

    def forward(self, x):
        self.lstm.flatten_parameters()
        # Set initial hidden and cell states
        h0 = torch.zeros(self.layer_num, x.size(0), self.hidden_size, dtype=torch.float64).to(DEVICE)
        c0 = torch.zeros(self.layer_num, x.size(0), self.hidden_size, dtype=torch.float64).to(DEVICE)

        out, (h_n, c_n) = self.lstm(x, (h0, c0)) # do not need (h_n, c_n)
        # x.shape = [batchSize, sequenceLEN, featureSIZE]
        # out.shape = [batchSize, sequenceLEN, hidden_num]
        # h_n.shape = c_n.shape = [layer_num, batchSize, hidden_num]
        out = out[:, -1, :]
        out = self.fully_connect(out)
        return out