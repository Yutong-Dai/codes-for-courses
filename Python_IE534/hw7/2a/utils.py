'''
File: utils.py
Author: Yutong Dai (rothdyt@gmail.com)
File Created: Tuesday, 2018-10-30 00:29
Last Modified: Tuesday, 2018-10-30 00:30
--------------------------------------------
Desscription: Helpfunctions and classes.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
import time
import os
import sys
import io
import logging

log_level = logging.INFO
logger = logging.getLogger()
logger.setLevel(log_level)
handler = logging.FileHandler("2a.log")
handler.setLevel(log_level)
formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class LockedDropout(nn.Module):
    def __init__(self):
        super(LockedDropout, self).__init__()
        self.m = None

    def reset_state(self):
        self.m = None

    def forward(self, x, dropout=0.5, train=True):
        if train == False:
            return x
        if(self.m is None):
            self.m = x.data.new(x.size()).bernoulli_(1 - dropout)
        mask = Variable(self.m, requires_grad=False) / (1 - dropout)

        return mask * x


def train(x_train, y_train, model, opt='adam', LR=0.001, batch_size=200, no_of_epochs=20, extension="ta"):
    logger.info("[Train] | Model:{} | embedding dimension:{} | optimizer:{} | learning rate:{} | epoch:{}".format(extension,
                                                                                                                  model.embedding.embedding_dim,
                                                                                                                  opt, LR, no_of_epochs))
    model.cuda()
    if (opt == 'adam'):
        optimizer = optim.Adam(model.parameters(), lr=LR)
    elif (opt == 'sgd'):
        optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, nesterov=True)

    L_Y_train = len(y_train)
    train_loss = []
    train_accu = []
    for epoch in range(no_of_epochs):
        epoch_acc = 0.0
        epoch_loss = 0.0
        epoch_counter = 0
        time1 = time.time()
        I_permutation = np.random.permutation(L_Y_train)
        for i in range(0, L_Y_train, batch_size):
            x_input2 = [x_train[j] for j in I_permutation[i:i+batch_size]]
            sequence_length = 100
            x_input = np.zeros((batch_size, sequence_length), dtype=np.int)
            for j in range(batch_size):
                x = np.asarray(x_input2[j])
                sl = x.shape[0]
                if(sl < sequence_length):
                    x_input[j, 0:sl] = x
                else:
                    start_index = np.random.randint(sl-sequence_length+1)
                    x_input[j, :] = x[start_index:(start_index+sequence_length)]
            y_input = np.asarray([y_train[j] for j in I_permutation[i:i+batch_size]], dtype=np.int)
            data = Variable(torch.LongTensor(x_input)).cuda()
            target = Variable(torch.FloatTensor(y_input)).cuda()
            optimizer.zero_grad()
            loss, pred = model(data, target, train=True)
            loss.backward()
            optimizer.step()   # update weights
            prediction = pred >= 0.0
            truth = target >= 0.5
            acc = prediction.eq(truth).sum().cpu().data.numpy()
            epoch_acc += acc
            epoch_loss += loss.data.item()
            epoch_counter += batch_size

        epoch_acc /= epoch_counter
        epoch_loss /= (epoch_counter/batch_size)
        train_loss.append(epoch_loss)
        train_accu.append(epoch_acc)

        logger.info("Epoch:{} | Train Accuracy:{} | Epoch Loss:{} | Time Elpased:{}".format(epoch, epoch_acc*100.0,  epoch_loss, float(time.time()-time1)))

    torch.save(model, './results/RNN_{}_{}.model'.format(extension, model.embedding.embedding_dim))
    data = [train_loss, train_accu]
    data = np.asarray(data)
    np.save('./results/data_train_{}_{}.npy'.format(extension, model.embedding.embedding_dim), data)


def test(x_test, y_test, model, opt='adam', LR=0.001, batch_size=200, no_of_test=9, extension="ta"):
    logger.info("[Test] | Model:{} | embedding dimension:{} | optimizer:{} | learning rate:{}".format(extension,
                                                                                                      model.embedding.embedding_dim,
                                                                                                      opt, LR))
    model.cuda()
    L_Y_test = len(y_test)
    test_accu = []
    for epoch in range(no_of_test):
        epoch_acc = 0.0
        epoch_loss = 0.0
        epoch_counter = 0
        time1 = time.time()
        I_permutation = np.random.permutation(L_Y_test)
        for i in range(0, L_Y_test, batch_size):
            x_input2 = [x_test[j] for j in I_permutation[i:i+batch_size]]
            sequence_length = (epoch+1)*50
            x_input = np.zeros((batch_size, sequence_length), dtype=np.int)
            for j in range(batch_size):
                x = np.asarray(x_input2[j])
                sl = x.shape[0]
                if(sl < sequence_length):
                    x_input[j, 0:sl] = x
                else:
                    start_index = np.random.randint(sl-sequence_length+1)
                    x_input[j, :] = x[start_index:(start_index+sequence_length)]
            y_input = np.asarray([y_test[j] for j in I_permutation[i:i+batch_size]], dtype=np.int)
            data = Variable(torch.LongTensor(x_input)).cuda()
            target = Variable(torch.FloatTensor(y_input)).cuda()
            with torch.no_grad():
                loss, pred = model(data, target, train=False)
            prediction = pred >= 0.0
            truth = target >= 0.5
            acc = prediction.eq(truth).sum().cpu().data.numpy()
            epoch_acc += acc
            epoch_loss += loss.data.item()
            epoch_counter += batch_size

        epoch_acc /= epoch_counter
        epoch_loss /= (epoch_counter/batch_size)
        test_accu.append(epoch_acc)
        time2 = time.time()
        time_elapsed = time2 - time1
        logger.info("seq_len:{} | Test Accuracy:{} | Epoch Loss:{} | Time Elpased:{}".format(sequence_length, epoch_acc*100.0,  epoch_loss, float(time_elapsed)))

    data = [test_accu]
    data = np.asarray(data)
    np.save('./results/data_test_{}_{}.npy'.format(extension, model.embedding.embedding_dim), data)
