"""
Usage:
    python hw4_transfer_learning.py --num_epochs 10 --batch_size 100 --test_only --resume './tf_checkpoint.pth.tar'
Reference:
[1] https://towardsdatascience.com/transfer-learning-using-pytorch-4c3475f4495
"""
import torchvision
import torchvision.transforms as transforms
import torch
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import os
import argparse
import logging
import numpy as np

from myutils import save_checkpoint, update_learning_rate
# General setups
parser = argparse.ArgumentParser(description='PyTorch CIFAR-100 Training')
parser.add_argument('--num_epochs', '-e', default=10, type=int, help='total training epoch')
parser.add_argument('--batch_size', '-b', default=100, type=int, help='batch size')
parser.add_argument('--resume', '-r', default="./checkpoint.pth.tar", help='resume from checkpoint')
parser.add_argument('--test_only', '-t', action='store_true', help='do not use it if you are not the devloper.')
args = parser.parse_args()

log_level = logging.INFO
logger = logging.getLogger()
logger.setLevel(log_level)
handler = logging.FileHandler("tf.log")
handler.setLevel(log_level)
formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info("torch version: {}".format(torch.__version__))

# from torch.utils import model_zoo
# model_urls = {'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth'}
# model = torchvision.models.resnet.ResNet(torchvision.models.resnet.BasicBlock, [2,2,2,2])
# model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir='./'))


# Hyper Parameters
batch_size = args.batch_size
# Data Preparation

# note that mean and std is calculated channel-wise
# reference: https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457/10
print("Data Preparation...")
logger.info("Data Preparation...")
transform_train = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.RandomCrop(224, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
])

transform_test = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
])

train_dataset = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR100(root='../data', train=False, download=False, transform=transform_test)
if args.test_only:
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=False, sampler=SubsetRandomSampler(range(2000)))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2,
                                              sampler=SubsetRandomSampler(range(100)))

print("Model setting...")
logger.info("Model setting...")

use_cuda = torch.cuda.is_available()
start_epoch = 0
net = torchvision.models.resnet.ResNet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2])
if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    print(torch.cuda.device_count())
    cudnn.benchmark = True
net.load_state_dict(torch.load("../data/model/resnet18-5c106cde.pth"))

# Do not change the layers that are pre-trained with the only exception
# on the last full-connected layer.
for param in net.parameters():
    param.requires_grad = False
# change the last fc layer for cifar100
net.fc = nn.Linear(in_features=net.fc.in_features, out_features=100)

#optimizer = optim.Adam(net.parameters())
optimizer = optim.SGD(net.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()
training_accuracy_seq = []
training_loss_seq = []
testing_accuracy_seq = []
testing_loss_seq = []
testing_best_accuracy = 0

if args.resume:
    print("Resume from the checkpoint...")
    logger.info("Resume from the checkpoint...")
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch'] + 1
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        training_accuracy_seq = checkpoint['training_accuracy_seq']
        training_loss_seq = checkpoint['training_loss_seq']
        testing_accuracy_seq = checkpoint['testing_accuracy_seq']
        testing_loss_seq = checkpoint['testing_loss_seq']
        testing_best_accuracy = checkpoint['testing_best_accuracy']
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, (checkpoint['epoch'] + 1)))
        logger.info("=> loaded checkpoint '{}' (epoch {})"
                    .format(args.resume, (checkpoint['epoch'] + 1)))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
        logger.info("=> no checkpoint found at '{}'".format(args.resume))
        print("=> Training based on the resnet-18 from scratch...")
        logger.info("=> Training based on the resnet-18 from scratch...")
else:
    print("=> Training based on the resnet-18 from scratch...")
    logger.info("=> Training based on the resnet-18 from scratch...")


print("Model Training...")
logger.info("Model Training...")

# use up-to-date learning rate; for resume purpose
for param_group in optimizer.param_groups:
    current_learningRate = param_group['lr']


def train(epoch):
    global current_learningRate
    net.train()
    if (epoch+1) % 20 == 0:
        current_learningRate /= 10
        logger.info("=> Learning rate is updated!")
        update_learning_rate(optimizer, current_learningRate)
    train_accuracy = []
    for _, (images, labels) in enumerate(train_loader):
        if use_cuda:
            images, labels = images.cuda(), labels.cuda()
        optimizer.zero_grad()
        images, labels = Variable(images), Variable(labels)
        output = net(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        predicted = output.data.max(1)[1]
        accuracy = (float(predicted.eq(labels.data).sum()) / float(batch_size))
        train_accuracy.append(accuracy)

    train_accuracy_epoch = np.mean(train_accuracy)
    if torch.__version__ == '0.4.1':
        loss_epoch = loss.item()
    else:
        loss_epoch = loss.data[0]

    print("=> Epoch: [{}/{}] | Loss:[{}] | Training Accuracy: [{}]".format(
        epoch + 1, args.num_epochs, loss_epoch, train_accuracy_epoch))
    logger.info("=> Epoch: [{}/{}] | Training Loss:[{}] | Training Accuracy: [{}]".format(
        epoch + 1, args.num_epochs, loss_epoch, train_accuracy_epoch))
    return loss_epoch, train_accuracy_epoch


def test(epoch):
    net.eval()
    test_accuracy = []
    for _, (images, labels) in enumerate(test_loader):
        if use_cuda:
            images, labels = images.cuda(), labels.cuda()
        images, labels = Variable(images), Variable(labels)
        output = net(images)
        loss = criterion(output, labels)
        predicted = output.data.max(1)[1]
        accuracy = (float(predicted.eq(labels.data).sum()) / float(100))  # test batch size 100
        test_accuracy.append(accuracy)

    test_accuracy_epoch = np.mean(test_accuracy)
    if torch.__version__ == '0.4.1':
        test_loss_epoch = loss.item()
    else:
        test_loss_epoch = loss.data[0]
    if (epoch + 1) % 5 == 0:
        print("=> Epoch: [{}/{}] | Loss:[{}] | Testing Accuracy: [{}]".format(
            epoch + 1, args.num_epochs, test_loss_epoch, test_accuracy_epoch))
        logger.info("=> Epoch: [{}/{}] | Testing Loss:[{}] | Testing Accuracy: [{}]".format(
            epoch + 1, args.num_epochs, test_loss_epoch, test_accuracy_epoch))

    return test_loss_epoch, test_accuracy_epoch


for epoch in range(start_epoch, args.num_epochs):
    train_loss, train_accuracy = train(epoch)
    test_loss, test_accuracy = test(epoch)

    training_loss_seq.append(train_loss)
    training_accuracy_seq.append(train_accuracy)
    testing_loss_seq.append(test_loss)
    testing_accuracy_seq.append(test_accuracy)

    is_best = testing_accuracy_seq[-1] > testing_best_accuracy
    testing_best_accuracy = max(testing_best_accuracy, testing_accuracy_seq[-1])

    state = {
        "epoch": epoch,
        "state_dict": net.state_dict(),  # if use_cuda else net.module.state_dict()
        "optimizer": optimizer.state_dict(),
        "training_loss_seq": training_loss_seq,
        "training_accuracy_seq": training_accuracy_seq,
        "testing_loss_seq": testing_loss_seq,
        "testing_accuracy_seq": testing_accuracy_seq,
        "testing_best_accuracy": testing_best_accuracy
    }
    save_checkpoint(state, is_best, filename='checkpoint.pth.tar', extra="tf_")
    if is_best:
        logger.info("=> Best parameters are updated")


logger.info("=> Trained on [{}] epoch, with test accuracy [{}].\n \
 During the training stages, historical best test accuracy is \
 [{}]".format(args.num_epochs, testing_accuracy_seq[-1], testing_best_accuracy))
print("=> Trained on [{}] epoch, with test accuracy [{}].\n \
 During the training stages, historical best test accuracy is \
 [{}]".format(args.num_epochs, testing_accuracy_seq[-1], testing_best_accuracy))
