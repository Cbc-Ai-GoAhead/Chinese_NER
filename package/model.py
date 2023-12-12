# -*- coding: utf-8 -*-
from torch import nn
from package.nn import ConditionalRandomField
import os
import torch


class BiLSTM_CRF(nn.Module):

    def __init__(self, tag_set_size, lstm_hidden_dim=256, lstm_dropout_rate=0.1):
        super(BiLSTM_CRF, self).__init__()

        self.bilstm = nn.LSTM(300, lstm_hidden_dim // 2,
                              num_layers=2, bidirectional=True, dropout=lstm_dropout_rate, batch_first=True)
        self.hidden2tag = nn.Linear(lstm_hidden_dim, tag_set_size)
        self.crf = ConditionalRandomField(tag_set_size)

        self.lstm_hidden_dim = lstm_hidden_dim

    def reset_parameters(self):
        self.crf.reset_parameters()

    def loss(self, input: torch.LongTensor, mask: torch.ByteTensor):

        x = input.float()
        x, _ = self.bilstm(x)
        x = self.hidden2tag(x)

        return self.crf(x, mask)

    def forward(self, input: torch.LongTensor, mask: torch.ByteTensor, target: torch.LongTensor):
        x = input.float()
        x, _ = self.bilstm(x)
        x = self.hidden2tag(x)
        return self.crf.neg_log_likelihood_loss(x,mask, target)

class BiLSTM_CRF_01_dropout(nn.Module):

    def __init__(self, tag_set_size, lstm_hidden_dim=256, lstm_dropout_rate=0.1):
        super(BiLSTM_CRF_01_dropout, self).__init__()

        self.bilstm = nn.LSTM(300, lstm_hidden_dim // 2,
                              num_layers=2, bidirectional=True, dropout=lstm_dropout_rate, batch_first=True)
        self.dropout_layer = nn.Dropout(0.5)
        self.hidden2tag = nn.Linear(lstm_hidden_dim, tag_set_size)
        self.crf = ConditionalRandomField(tag_set_size)

        self.lstm_hidden_dim = lstm_hidden_dim

    def reset_parameters(self):
        self.crf.reset_parameters()

    def loss(self, input: torch.LongTensor, mask: torch.ByteTensor):

        x = input.float()
        x, _ = self.bilstm(x)
        x = self.dropout_layer(x)
        x = self.hidden2tag(x)

        return self.crf(x, mask)

    def forward(self, input: torch.LongTensor, mask: torch.ByteTensor, target: torch.LongTensor):
        x = input.float()
        x, _ = self.bilstm(x)
        x = self.dropout_layer(x)
        x = self.hidden2tag(x)
        return self.crf.neg_log_likelihood_loss(x,mask, target)
class GRU_CRF(nn.Module):

    def __init__(self, tag_set_size, lstm_hidden_dim=256, lstm_dropout_rate=0.1):
        super(GRU_CRF, self).__init__()
        embedding_size = 300
        bidirectional = True
        self.gru = torch.nn.GRU(
            embedding_size,
            lstm_hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            dropout=lstm_dropout_rate,
            bidirectional=bidirectional
        )
        self.dropout_layer = nn.Dropout(0.5)
        self.hidden2tag = nn.Linear(lstm_hidden_dim, tag_set_size)
        self.crf = ConditionalRandomField(tag_set_size)

        self.lstm_hidden_dim = lstm_hidden_dim

    def reset_parameters(self):
        self.crf.reset_parameters()

    def loss(self, input: torch.LongTensor, mask: torch.ByteTensor):

        x = input.float()
        x, _ = self.gru(x)
        x = self.dropout_layer(x)
        x = self.hidden2tag(x)

        return self.crf(x, mask)

    def forward(self, input: torch.LongTensor, mask: torch.ByteTensor, target: torch.LongTensor):
        x = input.float()
        x, _ = self.gru(x)
        x = self.dropout_layer(x)
        x = self.hidden2tag(x)
        return self.crf.neg_log_likelihood_loss(x,mask, target)