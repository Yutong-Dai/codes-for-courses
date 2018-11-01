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

from RNN_model import RNN_model
from utils import test

glove_embeddings = np.load('../preprocessed_data/glove_embeddings.npy')
vocab_size = 100000

x_test = []
with io.open('../preprocessed_data/imdb_test_glove.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
for line in lines:
    line = line.strip()
    line = line.split(' ')
    line = np.asarray(line, dtype=np.int)
    line[line > vocab_size] = 0
    x_test.append(line)
y_test = np.zeros((25000,))
y_test[0:12500] = 1

# for unknown token
vocab_size += 1

extension, embedding_dim, opt = "ta", "500", "adam"
model = torch.load('./results/RNN_{}_{}_{}.model'.format(extension, embedding_dim, opt))
test(x_test, y_test, glove_embeddings, model, opt, LR=0.001, batch_size=200, no_of_test=9, extension="ta")

extension, embedding_dim, opt = "ta", "100", "adam"
model = torch.load('./results/RNN_{}_{}_{}.model'.format(extension, embedding_dim, opt))
test(x_test, y_test, glove_embeddings, model, opt, LR=0.001, batch_size=200, no_of_test=9, extension="ta")

extension, embedding_dim, opt = "ta", "2000", "adam"
model = torch.load('./results/RNN_{}_{}_{}.model'.format(extension, embedding_dim, opt))
test(x_test, y_test, glove_embeddings, model, opt, LR=0.001, batch_size=100, no_of_test=9, extension="ta")

extension, embedding_dim, opt = "ta", "500", "sgd"
model = torch.load('./results/RNN_{}_{}_{}.model'.format(extension, embedding_dim, opt))
test(x_test, y_test, glove_embeddings, model, opt, LR=0.001, batch_size=200, no_of_test=9, extension="ta")
