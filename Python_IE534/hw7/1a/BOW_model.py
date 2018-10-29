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
    def __init__(self, vocab_size, no_of_hidden_units):
        super(BOW_model_ta, self).__init__()
        # create a vocab_size * no_of_hidden_units weight matrix
        self.embedding = nn.Embedding(vocab_size, no_of_hidden_units)
        self.fc_hidden = nn.Linear(no_of_hidden_units, no_of_hidden_units)
        self.bn_hidden = nn.BatchNorm1d(no_of_hidden_units)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.fc_output = nn.Linear(no_of_hidden_units, 1)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x, t):
        """
        Assume the input `x` is a list of length `batch_size` and each element of this list is a numpy array 
        containing the token ids for a particular sequence. These sequences are different length which is why `x` 
        is not simply a torch tensor of size `batch_size` by `sequence_length`. 

        Within the loop, the `lookup_tensor` is a single sequence of token ids which can be fed into the embedding layer. 
        This returns a torch tensor of length `sequence_length` by `embedding_size`. 
        We take the mean over the dimension corresponding to the sequence length and append it to the list bow_embedding. 
        This mean operation is considered the bag of words. Note this operation returns the same vector `embed` regardless of 
        how the token ids were ordered in the `lookup_tensor`.
        """
        bow_embedding = []
        for i in range(len(x)):
            lookup_tensor = Variable(torch.LongTensor(x[i])).cuda()
            # embed is of size len(x[i]) by no_of_hidden_units
            embed = self.embedding(lookup_tensor)
            # seq_embed, the embedding for the whole sequence
            seq_embed = embed.mean(dim=0)
            bow_embedding.append(seq_embed)
        # batch_size by no_of_hidden_units vector
        bow_embedding = torch.stack(bow_embedding)

        h = self.dropout(F.relu(self.bn_hidden(self.fc_hidden(bow_embedding))))
        # batch_size by 1 vector
        h = self.fc_output(h)

        return self.loss(h[:, 0], t), h[:, 0]


class BOW_model_overfit(nn.Module):
    def __init__(self, vocab_size, no_of_hidden_units):
        super(BOW_model_overfit, self).__init__()
        # create a vocab_size * no_of_hidden_units weight matrix
        self.embedding = nn.Embedding(vocab_size, no_of_hidden_units)
        self.fc_hidden = nn.Linear(no_of_hidden_units, no_of_hidden_units)
        self.bn_hidden = nn.BatchNorm1d(no_of_hidden_units)
        self.fc_output = nn.Linear(no_of_hidden_units, 1)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x, t):
        bow_embedding = []
        for i in range(len(x)):
            lookup_tensor = Variable(torch.LongTensor(x[i])).cuda()
            # embed is of size len(x[i]) by no_of_hidden_units
            embed = self.embedding(lookup_tensor)
            # seq_embed, the embedding for the whole sequence
            seq_embed = embed.mean(dim=0)
            bow_embedding.append(seq_embed)
        # batch_size by no_of_hidden_units vector
        bow_embedding = torch.stack(bow_embedding)

        h = F.relu(self.bn_hidden(self.fc_hidden(bow_embedding)))
        # batch_size by 1 vector
        h = self.fc_output(h)

        return self.loss(h[:, 0], t), h[:, 0]
