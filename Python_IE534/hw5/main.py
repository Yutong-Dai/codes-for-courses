'''
File: main.py
Author: Yutong Dai (rothdyt@gmail.com)
File Created: Wednesday, 2018-10-10 13:02
Last Modified: Monday, 2018-10-15 01:29
--------------------------------------------
Desscription: hw5 (Learning Fine-grained Image Similarity with Deep Ranking).

python main.py --num_epochs 30 --batch_size 200 --train_all --resume './hw5_checkpoint.pth.tar'
python main.py --num_epochs 2 --batch_size 100 --train_all --resume './hw5_checkpoint.pth.tar' --test_only
'''

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
import pickle

import utils
# General setups
parser = argparse.ArgumentParser(description='PyTorch TinyImageNet Training 200 classes')
parser.add_argument('--num_epochs', '-e', default=10, type=int, help='total training epoch')
parser.add_argument('--batch_size', '-b', default=100, type=int, help='batch size')
parser.add_argument('--train_all', '-a', action='store_true', help='train all layers or only the last fc layer')
parser.add_argument('--resume', '-r', default="./checkpoint.pth.tar", help='resume from checkpoint')
parser.add_argument('--test_only', '-t', action='store_true', help='do not use it if you are not the devloper.')
args = parser.parse_args()

log_level = logging.INFO
logger = logging.getLogger()
logger.setLevel(log_level)
handler = logging.FileHandler("hw5.log")
handler.setLevel(log_level)
formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info("torch version: {}".format(torch.__version__))


# Hyper Parameters
batch_size = args.batch_size
topk = 30
pdist = nn.PairwiseDistance(p=2)
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
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_test = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


print("Loading Data...")
logger.info("Loading Data...")
img, label = utils.generate_testing_data_set()
test_dataset = utils.TinyImageNet(img, label, train=False, transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=8)

if args.test_only:
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=2,
                                              sampler=SubsetRandomSampler(range(8)))
print("Model setting...")
logger.info("Model setting...")

use_cuda = torch.cuda.is_available()
start_epoch = 0
net = torchvision.models.resnet.ResNet(torchvision.models.resnet.Bottleneck, [3, 4, 23, 3])
net.load_state_dict(torch.load("../data/model/resnet101-5d3b4d8f.pth"))

# Do not change the layers that are pre-trained with the only exception
# on the last full-connected layer.
if not args.train_all:
    for param in net.parameters():
        param.requires_grad = False
# change the last fc layer for cifar100
net.fc = nn.Linear(in_features=net.fc.in_features, out_features=4096)

#optimizer = optim.Adam(net.parameters())
optimizer = optim.SGD(net.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)

criterion = nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-06)
training_loss_seq = []
training_accuracy_seq = []
testing_accuracy_seq = []
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
        training_loss_seq = checkpoint['training_loss_seq']
        training_accuracy_seq = checkpoint['training_accuracy_seq']
        testing_accuracy_seq = checkpoint['testing_accuracy_seq']
        testing_best_accuracy = checkpoint['testing_best_accuracy']
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, (checkpoint['epoch'] + 1)))
        logger.info("=> loaded checkpoint '{}' (epoch {})"
                    .format(args.resume, (checkpoint['epoch'] + 1)))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
        logger.info("=> no checkpoint found at '{}'".format(args.resume))
        print("=> Training based on the resnet-101 from scratch...")
        logger.info("=> Training based on the resnet-101 from scratch...")
else:
    print("=> Training based on the resnet-18 from scratch...")
    logger.info("=> Training based on the resnet-18 from scratch...")


print("Model Training...")
logger.info("Model Training...")

# use up-to-date learning rate; for resume purpose
for param_group in optimizer.param_groups:
    current_learningRate = param_group['lr']

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    print(torch.cuda.device_count())
    cudnn.benchmark = True


def train(epoch):
    if args.test_only:
        topk = 3
        img_triplet, label_triplet = pickle.load(open("./pickle/train_1.p", 'rb'))
        train_dataset = utils.TinyImageNet(img_triplet, label_triplet, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=5, num_workers=2,
                                                   shuffle=False, sampler=SubsetRandomSampler([0, 503, 1003, 1503, 2003, 2503, 3003, 3503, 4003, 4503]))
    else:
        if not os.path.isfile("./pickle/train_{}.p".format(epoch)):
            img_triplet, label_triplet = utils.generate_training_data_set(save=True, epoch_idx=epoch)
        else:
            img_triplet, label_triplet = pickle.load(open("./pickle/train_{}.p".format(epoch), 'rb'))
        train_dataset = utils.TinyImageNet(img_triplet, label_triplet, train=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    global current_learningRate
    net.train()
    if (epoch+1) % 10 == 0:
        current_learningRate /= 2
        logger.info("=> Learning rate is updated!")
        utils.update_learning_rate(optimizer, current_learningRate)

    f_img_train = []
    label_train = []
    train_bacth_accuracy = []
    loss_epoch = 0
    for _, (images, lables) in enumerate(train_loader):
        if use_cuda:
            q, p, n, q_label = images[0].cuda(), images[1].cuda(), images[2].cuda(), lables[0].cuda()
        else:
            q, p, n, q_label = images[0], images[1], images[2], lables[0]
        optimizer.zero_grad()
        q, p, n = Variable(q), Variable(p), Variable(n)
        f_q, f_p, f_n = net(q), net(p), net(n)
        loss = criterion(f_q, f_p, f_n)
        loss.backward()
        optimizer.step()

        if torch.__version__ == '0.4.1':
            loss_epoch += loss.item()
        else:
            loss_epoch += loss.data[0]
        f_img_train.append(f_q)
        label_train.append(q_label)

        # calculate train_acc so use train_loader as the test_loader
        train_accuracy = []
        for f_img_test_current, label_test_current in zip(f_q, q_label):
            f_img_test_current = f_img_test_current.reshape(1, 4096)
            f_img_test_current = f_img_test_current.expand(f_q.shape[0], 4096)
            distance = pdist(f_img_test_current, f_q)
            predicted = q_label[distance.topk(topk)[1]]
            train_accuracy.append(float(torch.sum(torch.eq(predicted, label_test_current))) / topk)
        train_bacth_accuracy.append(np.mean(train_accuracy))
    train_accuracy_epoch = np.mean(train_bacth_accuracy)

    f_img_train = torch.cat(f_img_train, dim=0)
    label_train = torch.cat(label_train, dim=0)

    # train_accuracy = []
    # # calculate train_acc so use train_loader as the test_loader
    # for f_img_test_current, label_test_current in zip(f_img_train, label_train):
    #     f_img_test_current = f_img_test_current.reshape(1, 4096)
    #     f_img_test_current = f_img_test_current.expand(f_img_train.shape[0], 4096)
    #     distance = pdist(f_img_test_current, f_img_train)
    #     predicted = label_train[distance.topk(topk)[1]]
    #     train_accuracy.append(float(torch.sum(torch.eq(predicted, label_test_current))) / topk)
    # train_accuracy_epoch = np.mean(train_accuracy)

    print("=> Epoch: [{}/{}] | Loss:[{}] | Training Accuracy: [{}]".format(epoch + 1, args.num_epochs, loss_epoch, train_accuracy_epoch))
    logger.info("=> Epoch: [{}/{}] | Loss:[{}] | Training Accuracy: [{}]".format(epoch + 1, args.num_epochs, loss_epoch, train_accuracy_epoch))
    return loss_epoch, train_accuracy_epoch, f_img_train, label_train


def test(epoch, f_img_train, label_train):
    net.eval()
    if args.test_only:
        topk = 3
    # f_img_train = []
    # label_train = []
    # for _, (imgs_train, labels_train) in enumerate(train_loader):
    #     if use_cuda:
    #         imgs_train, labels_train = imgs_train.cuda(), labels_train.cuda()
    #     f_img_train.append(net(imgs_train[0]))
    #     label_train.append(labels_train[0])
    # f_img_train = torch.cat(f_img_train, dim=0)
    # label_train = torch.cat(label_train, dim=0)

    f_img_test = []
    label_test = []
    for _, (imgs_test, labels_test) in enumerate(test_loader):
        if use_cuda:
            imgs_test, labels_test = imgs_test.cuda(), labels_test.cuda()
        #f_img_test, label_test = Variable(f_img_test), Variable(label_test)
        f_img_test.append(net(imgs_test))
        label_test.append(labels_test)

    f_img_test = torch.cat(f_img_test, dim=0)
    label_test = torch.cat(label_test, dim=0)

    test_accuracy = []
    for f_img_test_current, label_test_current in zip(f_img_test, label_test):
        f_img_test_current = f_img_test_current.reshape(1, 4096)
        f_img_test_current = f_img_test_current.expand(f_img_train.shape[0], 4096)
        distance = pdist(f_img_test_current, f_img_train)
        predicted = label_train[distance.topk(topk)[1]]
        test_accuracy.append(float(torch.sum(torch.eq(predicted, label_test_current))) / topk)
    test_accuracy_epoch = np.mean(test_accuracy)

    print("=> Epoch: [{}/{}] | Testing Accuracy: [{}]".format(
        epoch + 1, args.num_epochs, test_accuracy_epoch))
    logger.info("=> Epoch: [{}/{}] | Testing Accuracy: [{}]".format(
        epoch + 1, args.num_epochs, test_accuracy_epoch))

    return test_accuracy_epoch


for epoch in range(start_epoch, args.num_epochs):
    train_loss, train_accuracy, f_img_train, label_train = train(epoch)
    test_accuracy = test(epoch, f_img_train, label_train)

    training_loss_seq.append(train_loss)
    training_accuracy_seq.append(train_accuracy)
    testing_accuracy_seq.append(test_accuracy)

    is_best = testing_accuracy_seq[-1] > testing_best_accuracy
    testing_best_accuracy = max(testing_best_accuracy, testing_accuracy_seq[-1])

    state = {
        "epoch": epoch,
        "state_dict": net.state_dict(),  # if use_cuda else net.module.state_dict()
        "optimizer": optimizer.state_dict(),
        "training_loss_seq": training_loss_seq,
        "training_accuracy_seq": training_accuracy_seq,
        "testing_accuracy_seq": testing_accuracy_seq,
        "testing_best_accuracy": testing_best_accuracy
    }
    utils.save_checkpoint(state, is_best, filename='checkpoint.pth.tar', extra="hw5_")
    if is_best:
        logger.info("=> Best parameters are updated")


logger.info("=> Trained on [{}] epoch, with test accuracy [{}].\n \
 During the training stages, historical best test accuracy is \
 [{}]".format(args.num_epochs, testing_accuracy_seq[-1], testing_best_accuracy))
print("=> Trained on [{}] epoch, with test accuracy [{}].\n \
 During the training stages, historical best test accuracy is \
 [{}]".format(args.num_epochs, testing_accuracy_seq[-1], testing_best_accuracy))
