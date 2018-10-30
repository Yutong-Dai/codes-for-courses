'''
File: BOW_model.py
Author: Yutong Dai (rothdyt@gmail.com)
File Created: Sunday, 2018-10-28 23:22
Last Modified: Sunday, 2018-10-28 23:22
--------------------------------------------
Desscription: Define the bag of words pytorch model.
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.distributed as dist


class BOW_model_ta(nn.Module):
    def __init__(self, no_of_hidden_units):
        super(BOW_model_ta, self).__init__()
        self.fc_hidden1 = nn.Linear(300, no_of_hidden_units)
        self.bn_hidden1 = nn.BatchNorm1d(no_of_hidden_units)
        self.dropout1 = torch.nn.Dropout(p=0.5)
        self.fc_output = nn.Linear(no_of_hidden_units, 1)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x, t):
        # the input is a batch-size by 300
        h = self.dropout1(F.relu(self.bn_hidden1(self.fc_hidden1(x))))
        h = self.fc_output(h)
        return self.loss(h[:, 0], t), h[:, 0]


class BOW_model_overfit(nn.Module):
    def __init__(self, no_of_hidden_units):
        super(BOW_model_overfit, self).__init__()
        self.fc_hidden1 = nn.Linear(300, no_of_hidden_units)
        self.bn_hidden1 = nn.BatchNorm1d(no_of_hidden_units)
        self.fc_hidden2 = nn.Linear(no_of_hidden_units, no_of_hidden_units)
        self.bn_hidden2 = nn.BatchNorm1d(no_of_hidden_units)
        self.fc_output = nn.Linear(no_of_hidden_units, 1)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x, t):
        # the input is a batch-size by 300
        h = F.relu(self.bn_hidden1(self.fc_hidden1(x)))
        h = F.relu(self.bn_hidden2(self.fc_hidden2(h)))
        h = self.fc_output(h)
        return self.loss(h[:, 0], t), h[:, 0]
