'''
File: RNN_train.py
Author: Yutong Dai (rothdyt@gmail.com)
File Created: Friday, 2018-11-09 00:37
Last Modified: Friday, 2018-11-09 10:06
--------------------------------------------
Desscription:
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import time
import os
import sys

from utils import train
from RNN_language_model import RNN_language_model

imdb_dictionary = np.load('../preprocessed_data/imdb_dictionary.npy')
vocab_size = 8000

x_train = []
with open('../preprocessed_data/imdb_train.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
for line in lines:
    line = line.strip()
    line = line.split(' ')
    line = np.asarray(line, dtype=np.int)
    line[line > vocab_size] = 0
    x_train.append(line)

x_train = x_train[0:25000]
y_train = np.zeros((25000,))
y_train[0:12500] = 1


vocab_size += 1
batch_size = 200
no_of_epochs = 20
model = RNN_language_model(vocab_size, 500)
language_model = torch.load('language.model')
model.embedding.load_state_dict(language_model.embedding.state_dict())
model.lstm1.lstm.load_state_dict(language_model.lstm1.lstm.state_dict())
model.bn_lstm1.load_state_dict(language_model.bn_lstm1.state_dict())
model.lstm2.lstm.load_state_dict(language_model.lstm2.lstm.state_dict())
model.bn_lstm2.load_state_dict(language_model.bn_lstm2.state_dict())
model.lstm3.lstm.load_state_dict(language_model.lstm3.lstm.state_dict())
model.bn_lstm3.load_state_dict(language_model.bn_lstm3.state_dict())


train(x_train, y_train, model,  sequence_length=100, batch_size=200, no_of_epochs=20, train_layer="last", LR=0.001)
train(x_train, y_train, model,  sequence_length=50, batch_size=200, no_of_epochs=20, train_layer="last", LR=0.001)
train(x_train, y_train, model,  sequence_length=200, batch_size=100, no_of_epochs=20, train_layer="last", LR=0.001)
train(x_train, y_train, model,  sequence_length=100, batch_size=100, no_of_epochs=20, train_layer="all", LR=0.001)
