'''
File: BOW_sentiment_analysis.py
Author: Yutong Dai (rothdyt@gmail.com)
File Created: Monday, 2018-10-29 01:09
Last Modified: Monday, 2018-10-29 01:10
--------------------------------------------
Desscription:
'''


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.distributed as dist

import time
import os
import sys
import io

from BOW_model import BOW_model_ta, BOW_model_overfit
from utils import train_test

glove_embeddings = np.load('../preprocessed_data/glove_embeddings.npy')
vocab_size = 100000

x_train = []
with io.open('../preprocessed_data/imdb_train_glove.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
for line in lines:
    line = line.strip()
    line = line.split(' ')
    line = np.asarray(line, dtype=np.int)

    line[line > vocab_size] = 0
    line = line[line != 0]

    line = np.mean(glove_embeddings[line], axis=0)

    x_train.append(line)
x_train = np.asarray(x_train)
x_train = x_train[0:25000]
y_train = np.zeros((25000,))
y_train[0:12500] = 1

x_test = []
with io.open('../preprocessed_data/imdb_test_glove.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
for line in lines:
    line = line.strip()
    line = line.split(' ')
    line = np.asarray(line, dtype=np.int)

    line[line > vocab_size] = 0
    line = line[line != 0]

    line = np.mean(glove_embeddings[line], axis=0)

    x_test.append(line)
x_test = np.asarray(x_test)
y_test = np.zeros((25000,))
y_test[0:12500] = 1

vocab_size += 1

# Adam
model = BOW_model_ta(500)
train_test(x_train, x_test, y_train, y_test, model, opt='adam', LR=0.001, batch_size=200, no_of_epochs=8, extension="ta")

model = BOW_model_ta(300)
train_test(x_train, x_test, y_train, y_test, model, opt='adam', LR=0.001, batch_size=200, no_of_epochs=8, extension="under")

model = BOW_model_overfit(2000)
train_test(x_train, x_test, y_train, y_test, model, opt='adam', LR=0.001, batch_size=100, no_of_epochs=8, extension="over")

# SGD
model = BOW_model_ta(500)
train_test(x_train, x_test, y_train, y_test, model, opt='sgd', LR=0.001, batch_size=200, no_of_epochs=8, extension="ta")
