"""
Implementation of Resnet.
    1. The whole script is structed to follow the style suggested by reference [1].
    2. The network(Resnet) is loaded from the network.py.
    3. Features:
        * You can resume the training by passing the argument --resume followed by the address of the checkpoint file.
        * You can continue training by adding the epochs (--num_epochs some_number_larger_than_previous_setting)

Example:
In terminal, 
    python hw4_bw.py --num_epochs 10 --batch_size 100 --test_only --resume './checkpoint.pth.tar'
    python hw4_bw.py --num_epochs 50 --batch_size 500 --resume './checkpoint.pth.tar'
Reference:
1. https://github.com/meliketoy/wide-resnet.pytorch/blob/master/main.py 
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
from resnet import ResidualBlock, ResNet

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
handler = logging.FileHandler("task.log")
handler.setLevel(log_level)
formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info("torch version: {}".format(torch.__version__))


# Hyper Parameters
batch_size = args.batch_size

# Data Preparation

# note that mean and std is calculated channel-wise
# reference: https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457/10
print("Data Preparation...")
logger.info("Data Preparation...")
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
])

train_dataset = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR100(root='../data', train=False, download=False, transform=transform_test)

print("Loading Data...")
logger.info("Loading Data...")
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)

if args.test_only:
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=False, sampler=SubsetRandomSampler(range(2000)))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2,
                                              sampler=SubsetRandomSampler(range(100)))

print("Model setting...")
logger.info("Model setting...")

use_cuda = torch.cuda.is_available()
start_epoch = 0

net = ResNet()
optimizer = optim.Adam(net.parameters())
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
              .format(args.resume, checkpoint['epoch']))
        logger.info("=> loaded checkpoint '{}' (epoch {})"
                    .format(args.resume, (checkpoint['epoch'] + 1)))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
        logger.info("=> no checkpoint found at '{}'".format(args.resume))
        print("Training the resnet from scratch...")
        logger.info("Training the resnet from scratch...")
else:
    print("Training the resnet from scratch...")
    logger.info("Training the resnet from scratch...")

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    print(torch.cuda.device_count())
    cudnn.benchmark = True

print("Model Training...")
logger.info("Model Training...")

current_learningRate = 0.001


def train(epoch):
    net.train()
    if (epoch+1) % 20 == 0:
        current_learningRate /= 10
        print("learning rate is updated!")
        logger.info("learning rate is updated!")
        update_learning_rate(optimizer, current_learningRate)
    train_accuracy = []
    for batch_idx, (images, labels) in enumerate(train_loader):
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

    print("Epoch: [{}/{}] | Loss:[{}] | Training Accuracy: [{}]".format(
        epoch + 1, args.num_epochs, loss_epoch, train_accuracy_epoch))
    logger.info("Epoch: [{}/{}] | Training Loss:[{}] | Training Accuracy: [{}]".format(
        epoch + 1, args.num_epochs, loss_epoch, train_accuracy_epoch))
    return loss_epoch, train_accuracy_epoch


def test(epoch):
    net.eval()
    test_accuracy = []
    for batch_idx, (images, labels) in enumerate(test_loader):
        if use_cuda:
            images, labels = images.cuda(), labels.cuda()
        images, labels = Variable(images), Variable(labels)
        output = net(images)
        loss = criterion(output, labels)
        predicted = output.data.max(1)[1]
        accuracy = (float(predicted.eq(labels.data).sum()) / float(batch_size))
        test_accuracy.append(accuracy)

    test_accuracy_epoch = np.mean(test_accuracy)
    if torch.__version__ == '0.4.1':
        test_loss_epoch = loss.item()
    else:
        test_loss_epoch = loss.data[0]
    if (epoch + 1) % 5 == 0:
        print("Epoch: [{}/{}] | Loss:[{}] | Testing Accuracy: [{}]".format(
            epoch + 1, args.num_epochs, test_loss_epoch, test_accuracy_epoch))
        logger.info("Epoch: [{}/{}] | Testing Loss:[{}] | Testing Accuracy: [{}]".format(
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
    save_checkpoint(state, is_best)
    if is_best:
        logger.info("Best parameters are updated")


logger.info("Trained on [{}] epoch, with test accuracy [{}].\n \
 During the training stages, historical best test accuracy is \
 [{}]".format(args.num_epochs, testing_accuracy_seq[-1], testing_best_accuracy))
print("Trained on [{}] epoch, with test accuracy [{}].\n \
 During the training stages, historical best test accuracy is \
 [{}]".format(args.num_epochs, testing_accuracy_seq[-1], testing_best_accuracy))
