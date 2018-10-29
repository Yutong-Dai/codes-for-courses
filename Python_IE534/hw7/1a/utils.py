import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import time
import os
import sys
import io
import logging

log_level = logging.INFO
logger = logging.getLogger()
logger.setLevel(log_level)
handler = logging.FileHandler("1a.log")
handler.setLevel(log_level)
formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def train_test(x_train, x_test, y_train, y_test, model, opt='adam', LR=0.001, batch_size=200, no_of_epochs=6, extension="ta"):
    logger.info("BOW:{} | embedding dimension:{} |optimizer:{} | learning rate:{} | epoch:{}".format(extension,
                                                                                                     model.embedding.embedding_dim,
                                                                                                     opt, LR, no_of_epochs))
    model.cuda()
    if (opt == 'adam'):
        optimizer = optim.Adam(model.parameters(), lr=LR)
    elif (opt == 'sgd'):
        optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, nesterov=True)

    L_Y_train = len(y_train)
    L_Y_test = len(y_test)

    model.train()
    train_loss = []
    train_accu = []
    test_accu = []
    for epoch in range(no_of_epochs):
        model.train()
        epoch_acc = 0.0
        epoch_loss = 0.0
        epoch_counter = 0
        time1 = time.time()
        I_permutation = np.random.permutation(L_Y_train)
        for i in range(0, L_Y_train, batch_size):
            x_input = [x_train[j] for j in I_permutation[i:i+batch_size]]
            y_input = np.asarray([y_train[j] for j in I_permutation[i:i+batch_size]], dtype=np.int)
            target = Variable(torch.FloatTensor(y_input)).cuda()
            optimizer.zero_grad()
            loss, pred = model(x_input, target)
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

        model.eval()
        epoch_acc = 0.0
        epoch_loss = 0.0
        epoch_counter = 0
        time1 = time.time()
        I_permutation = np.random.permutation(L_Y_test)
        for i in range(0, L_Y_test, batch_size):
            x_input = [x_test[j] for j in I_permutation[i:i+batch_size]]
            y_input = np.asarray([y_test[j] for j in I_permutation[i:i+batch_size]], dtype=np.int)
            target = Variable(torch.FloatTensor(y_input)).cuda()
            with torch.no_grad():
                loss, pred = model(x_input, target)
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
        logger.info("Epoch:{} | Test Accuracy:{} | Epoch Loss:{} | Time Elpased:{}".format(epoch, epoch_acc*100.0,  epoch_loss, float(time_elapsed)))

    torch.save(model, './results/BOW_{}_{}.model'.format(extension, model.embedding.embedding_dim))
    data = [train_loss, train_accu, test_accu]
    data = np.asarray(data)
    np.save('./results/data_{}_{}.npy'.format(extension, model.embedding.embedding_dim), data)


"""
This should only take about 15-30 minutes to train. 
Report results for this model as well as at least two other trials (run more but only report on two interesting cases). 
Try to get a model that overfits (high accuracy on training data, lower on testing data) as well as one that underfits (low accuracy on both). 
Overfitting is easily achieved by greatly increasing the no_of_hidden_units, removing dropout, adding a second or third hidden layer in BOW_model.py as well as a few other things. 
Underfitting can be easily done by using significantly less hidden units.
"""
