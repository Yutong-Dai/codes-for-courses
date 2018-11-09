'''
File: utils.py
Author: Yutong Dai (rothdyt@gmail.com)
File Created: Friday, 2018-11-09 10:30
Last Modified: Friday, 2018-11-09 10:31
--------------------------------------------
Desscription:
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
handler = logging.FileHandler("3c.log")
handler.setLevel(log_level)
formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def train(x_train, y_train, model,  sequence_length=100, batch_size=200, no_of_epochs=20, train_layer="last", LR=0.001):
    logger.info("[Train] | train seq_length:{} | train_layer:{} | Epochs:{} | Batch Size:{}".format(sequence_length, train_layer, no_of_epochs, batch_size))
    model.cuda()
    params = []
    if (train_layer == "last"):
        for param in model.lstm3.parameters():
            params.append(param)
        for param in model.bn_lstm3.parameters():
            params.append(param)
        for param in model.fc_output.parameters():
            params.append(param)
    else:
        for param in model.embedding.parameters():
            params.append(param)
        for param in model.lstm1.parameters():
            params.append(param)
        for param in model.bn_lstm1.parameters():
            params.append(param)
        for param in model.lstm2.parameters():
            params.append(param)
        for param in model.bn_lstm2.parameters():
            params.append(param)
        for param in model.lstm3.parameters():
            params.append(param)
        for param in model.bn_lstm3.parameters():
            params.append(param)
        for param in model.fc_output.parameters():
            params.append(param)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    L_Y_train = len(y_train)
    train_loss = []
    train_accu = []
    for epoch in range(no_of_epochs):
        if (epoch+1) % 5 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = LR*0.8
        model.train()
        epoch_acc = 0.0
        epoch_loss = 0.0
        epoch_counter = 0
        time1 = time.time()
        I_permutation = np.random.permutation(L_Y_train)
        for i in range(0, L_Y_train, batch_size):
            x_input2 = [x_train[j] for j in I_permutation[i:i+batch_size]]
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

    torch.save(model, './results/RNN_{}_{}.model'.format(train_layer, sequence_length))
    data = [train_loss, train_accu]
    data = np.asarray(data)
    np.save('./results/data_train_{}_{}.npy'.format(train_layer, sequence_length), data)


def test(x_test, y_test, model,  train_layer, sequence_length, LR=0.001, batch_size=200, no_of_test=9):
    logger.info("[Test] | Model: train seq_length:{}, train_layer:{}".format(sequence_length, train_layer))
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
    np.save('./results/data_test_{}_{}.npy'.format(sequence_length, train_layer), data)
