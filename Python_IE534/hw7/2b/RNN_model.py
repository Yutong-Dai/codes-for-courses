
'''
File: RNN_model.py
Author: Yutong Dai (rothdyt@gmail.com)
File Created: Monday, 2018-10-29 20:30
Last Modified: Tuesday, 2018-10-30 00:06
--------------------------------------------
Desscription: Customized LSTM.
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from utils import LockedDropout


class StatefulLSTM(nn.Module):
    def __init__(self, in_size, out_size):
        super(StatefulLSTM, self).__init__()
        # Forget Gate, Input Gate, Output Gate are calculated within LSTMCell
        # out_size is the dimension of the hidden state and cell state
        self.lstm = nn.LSTMCell(in_size, out_size)
        self.out_size = out_size
        self.h = None
        self.c = None

    def reset_state(self):
        self.h = None
        self.c = None

    def forward(self, x):
        batch_size = x.data.size()[0]
        if self.h is None:
            state_size = [batch_size, self.out_size]
            # initialize cell state and hiddden state
            self.c = Variable(torch.zeros(state_size)).cuda()
            self.h = Variable(torch.zeros(state_size)).cuda()
        # update cell state and hiddden state
        self.h, self.c = self.lstm(x, (self.h, self.c))
        return self.h


class RNN_model(nn.Module):
    def __init__(self, no_of_hidden_units):
        super(RNN_model, self).__init__()
        self.lstm1 = StatefulLSTM(300, no_of_hidden_units)
        self.bn_lstm1 = nn.BatchNorm1d(no_of_hidden_units)
        self.dropout1 = LockedDropout()
        self.fc_output = nn.Linear(no_of_hidden_units, 1)
        self.loss = nn.BCEWithLogitsLoss()

    def reset_state(self):
        self.lstm1.reset_state()
        self.dropout1.reset_state()

    def forward(self, x, t, train=True):
        no_of_timesteps = x.shape[1]
        self.reset_state()
        outputs = []
        for i in range(no_of_timesteps):
            h = self.lstm1(x[:, i, :])
            h = self.bn_lstm1(h)
            h = self.dropout1(h, dropout=0.5, train=train)
            outputs.append(h)
        # (time_steps,batch_size,features)
        outputs = torch.stack(outputs)
        # (batch_size,features,time_steps)
        outputs = outputs.permute(1, 2, 0)
        pool = nn.MaxPool1d(no_of_timesteps)
        h = pool(outputs)
        # h: batch_size by no_of_hidden_units
        h = h.view(h.size(0), -1)
        h = self.fc_output(h)
        return self.loss(h[:, 0], t), h[:, 0]
