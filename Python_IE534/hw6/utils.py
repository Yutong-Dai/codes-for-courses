'''
File: utils.py
Author: Yutong Dai (rothdyt@gmail.com)
File Created: Thursday, 2018-10-23 17:25
Last Modified: Sunday, 2018-10-27 15:28
--------------------------------------------
Desscription: Helpfunctions.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.autograd import Variable


class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 196, 3, padding=1, stride=1,),
                                   nn.LeakyReLU(),
                                   nn.LayerNorm([196, 32, 32]))
        self.conv2 = nn.Sequential(nn.Conv2d(196, 196, 3, padding=1, stride=2),
                                   nn.LeakyReLU(),
                                   nn.LayerNorm([196, 16, 16]))
        self.conv3 = nn.Sequential(nn.Conv2d(196, 196, 3, padding=1, stride=1),
                                   nn.LeakyReLU(),
                                   nn.LayerNorm([196, 16, 16]))
        self.conv4 = nn.Sequential(nn.Conv2d(196, 196, 3, padding=1, stride=2),
                                   nn.LeakyReLU(),
                                   nn.LayerNorm([196, 8, 8]))
        self.conv5 = nn.Sequential(nn.Conv2d(196, 196, 3, padding=1, stride=1),
                                   nn.LeakyReLU(),
                                   nn.LayerNorm([196, 8, 8]))
        self.conv6 = nn.Sequential(nn.Conv2d(196, 196, 3, padding=1, stride=1),
                                   nn.LeakyReLU(),
                                   nn.LayerNorm([196, 8, 8]))
        self.conv7 = nn.Sequential(nn.Conv2d(196, 196, 3, padding=1, stride=1),
                                   nn.LeakyReLU(),
                                   nn.LayerNorm([196, 8, 8]))
        self.conv8 = nn.Sequential(nn.Conv2d(196, 196, 3, padding=1, stride=2),
                                   nn.LeakyReLU(),
                                   nn.LayerNorm([196, 4, 4]))
        self.maxpool = nn.MaxPool2d(4, padding=0, stride=4)
        self.fc1 = nn.Linear(in_features=196, out_features=1)
        self.fc10 = nn.Linear(in_features=196, out_features=10)

    def forward(self, x, extract_features=0):
        x = self.conv1(x)
        x = self.conv2(x)
        if(extract_features == 2):
            h = F.max_pool2d(x, 16, 16)
            h = h.view(-1, 196)
            return h
        x = self.conv3(x)
        x = self.conv4(x)
        if(extract_features == 4):
            h = F.max_pool2d(x, 8, 8)
            h = h.view(-1, 196)
            return h
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        if(extract_features == 8):
            h = F.max_pool2d(x, 4, 4)
            h = h.view(-1, 196)
            return h
        x = self.maxpool(x)
        x = x.view(-1, 196)
        fc1_output = self.fc1(x)
        fc10_output = self.fc10(x)
        return [fc1_output, fc10_output]


class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(100, 196*4*4),
                                 nn.BatchNorm1d(196*4*4))
        self.conv1 = nn.Sequential(nn.ConvTranspose2d(196, 196, 4, padding=1, stride=2),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(196))
        self.conv2 = nn.Sequential(nn.Conv2d(196, 196, kernel_size=3, padding=1, stride=1),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(196))
        self.conv3 = nn.Sequential(nn.Conv2d(196, 196, kernel_size=3, padding=1, stride=1),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(196))
        self.conv4 = nn.Sequential(nn.Conv2d(196, 196, kernel_size=3, padding=1, stride=1),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(196))
        self.conv5 = nn.Sequential(nn.ConvTranspose2d(196, 196, 4, padding=1, stride=2),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(196))
        self.conv6 = nn.Sequential(nn.Conv2d(196, 196, kernel_size=3, padding=1, stride=1),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(196))
        self.conv7 = nn.Sequential(nn.ConvTranspose2d(196, 196, 4, padding=1, stride=2),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(196))
        self.conv8 = nn.Conv2d(196, 3, 3, padding=1, stride=1)

    def forward(self, x):
        x = self.fc1(x)
        x = x.reshape(100, 196, 4, 4)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        output = torch.tanh(x)
        return output


def plot(samples):
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(10, 10)
    gs.update(wspace=0.02, hspace=0.02)
    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample)
    return fig


def plot_max_class(X_batch, model, label="without"):
    X = X_batch.mean(dim=0)
    X = X.repeat(10, 1, 1, 1)
    Y = torch.arange(10).type(torch.int64)
    Y = Variable(Y).cuda()
    lr = 0.1
    weight_decay = 0.001
    for i in range(200):
        output = model(X)
        output = output[1]
        loss = -output[torch.arange(10).type(torch.int64), torch.arange(10).type(torch.int64)]
        gradients = torch.autograd.grad(outputs=loss, inputs=X,
                                        grad_outputs=torch.ones(loss.size()).cuda(),
                                        create_graph=True, retain_graph=False, only_inputs=True)[0]

        prediction = output.data.max(1)[1]  # first column has actual prob.
        accuracy = (float(prediction.eq(Y.data).sum()) / float(10.0))*100.0
        print(i, accuracy, -loss)

        X = X - lr*gradients.data - weight_decay*X.data*torch.abs(X.data)
        X[X > 1.0] = 1.0
        X[X < -1.0] = -1.0

    # save new images
    samples = X.data.cpu().numpy()
    samples += 1.0
    samples /= 2.0
    samples = samples.transpose(0, 2, 3, 1)

    fig = plot(samples)
    plt.savefig('./visualization/max_class_{}_generator.png'.format(label), bbox_inches='tight')
    plt.close(fig)


def plot_max_feature(X_batch, model, extract_features, label="without", batch_size=100):
    lr = 0.1
    weight_decay = 0.001
    X = X_batch.mean(dim=0)
    X = X.repeat(batch_size, 1, 1, 1)
    Y = torch.arange(batch_size).type(torch.int64)
    Y = Variable(Y).cuda()
    for i in range(200):
        output = model(X, extract_features)
        loss = -output[torch.arange(batch_size).type(torch.int64), torch.arange(batch_size).type(torch.int64)]
        gradients = torch.autograd.grad(outputs=loss, inputs=X,
                                        grad_outputs=torch.ones(loss.size()).cuda(),
                                        create_graph=True, retain_graph=False, only_inputs=True)[0]

        prediction = output.data.max(1)[1]
        accuracy = (float(prediction.eq(Y.data).sum()) / float(batch_size))*100.0
        print(i, accuracy, -loss)

        X = X - lr*gradients.data - weight_decay*X.data*torch.abs(X.data)
        X[X > 1.0] = 1.0
        X[X < -1.0] = -1.0

    # save new images
    samples = X.data.cpu().numpy()
    samples += 1.0
    samples /= 2.0
    samples = samples.transpose(0, 2, 3, 1)

    fig = plot(samples[0:100])
    plt.savefig('./visualization/max_features_{}_generator_layer_{}.png'.format(label, extract_features), bbox_inches='tight')
    plt.close(fig)
