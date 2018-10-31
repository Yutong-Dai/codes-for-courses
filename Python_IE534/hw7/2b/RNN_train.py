'''
File: RNN_train.py
Author: Yutong Dai (rothdyt@gmail.com)
File Created: Monday, 2018-10-29 20:31
Last Modified: Tuesday, 2018-10-30 16:24
--------------------------------------------
Desscription:
'''
import numpy as np
import io
from RNN_model import RNN_model
from utils import train

vocab_size = 8000

x_train = []
with io.open('../preprocessed_data/imdb_train.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
for line in lines:
    line = line.strip()
    line = line.split(' ')
    line = np.asarray(line, dtype=np.int)
    # convert any token id greater than the dictionary size to the unknown token ID 0
    line[line > vocab_size] = 0
    x_train.append(line)

x_train = x_train[0:25000]
y_train = np.zeros((25000,))
# positive label
y_train[0:12500] = 1

# for unknown token
vocab_size += 1

# Adam
model = RNN_model(vocab_size, 500)
train(x_train, y_train, model, opt='adam', LR=0.001, batch_size=200, no_of_epochs=20, extension="ta")

model = RNN_model(vocab_size, 100)
train(x_train, y_train, model, opt='adam', LR=0.001, batch_size=200, no_of_epochs=20, extension="ta")

model = RNN_model(vocab_size=10000, no_of_hidden_units=500)
train(x_train, y_train, model, opt='adam', LR=0.001, batch_size=100, no_of_epochs=20, extension="ta")

# SGD
model = RNN_model(vocab_size, 500)
train(x_train, y_train, model, opt='sgd', LR=0.001, batch_size=200, no_of_epochs=20, extension="ta")
