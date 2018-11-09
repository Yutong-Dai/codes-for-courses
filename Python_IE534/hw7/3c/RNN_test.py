'''
File: RNN_test.py
Author: Yutong Dai (rothdyt@gmail.com)
File Created: Monday, 2018-10-29 20:31
Last Modified: Tuesday, 2018-10-30 18:09
--------------------------------------------
Desscription:
'''

import numpy as np
import torch
import io

from utils import test

imdb_dictionary = np.load('../preprocessed_data/imdb_dictionary.npy')
vocab_size = 8000
# for unknown token
vocab_size += 1

x_test = []
with io.open('../preprocessed_data/imdb_test.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
for line in lines:
    line = line.strip()
    line = line.split(' ')
    line = np.asarray(line, dtype=np.int)
    # convert any token id greater than the dictionary size to the unknown token ID 0
    line[line > vocab_size] = 0
    x_test.append(line)

y_test = np.zeros((25000,))
# positive label
y_test[0:12500] = 1

train_layer, sequence_length = "last", 100
model = torch.load('./results/RNN_{}_{}.model'.format(train_layer, sequence_length))
test(x_test, y_test, model,  train_layer, sequence_length, LR=0.001, batch_size=200, no_of_test=9)

train_layer, sequence_length = "last", 50
model = torch.load('./results/RNN_{}_{}.model'.format(train_layer, sequence_length))
test(x_test, y_test, model,  train_layer, sequence_length, LR=0.001, batch_size=200, no_of_test=9)

train_layer, sequence_length = "last", 200
model = torch.load('./results/RNN_{}_{}.model'.format(train_layer, sequence_length))
test(x_test, y_test, model,  train_layer, sequence_length, LR=0.001, batch_size=100, no_of_test=9)

train_layer, sequence_length = "all", 100
model = torch.load('./results/RNN_{}_{}.model'.format(train_layer, sequence_length))
test(x_test, y_test, model,  train_layer, sequence_length, LR=0.001, batch_size=100, no_of_test=9)
