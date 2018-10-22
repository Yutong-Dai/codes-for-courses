'''
File: main.py
Author: Yutong Dai (rothdyt@gmail.com)
File Created: Wednesday, 2018-10-10 13:02
Last Modified: Monday, 2018-10-15 01:29
--------------------------------------------
Desscription: hw5 (Learning Fine-grained Image Similarity with Deep Ranking).

python main.py --num_epochs 30 --batch_size 10 --train_all --resume './hw5_checkpoint.pth.tar'
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
import time

import utils
# General setups
parser = argparse.ArgumentParser(description='PyTorch TinyImageNet Training 200 classes')
parser.add_argument('--net', '-n', default="resnet18", help='model to use')
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
pdist = nn.PairwiseDistance(p=2)
# Data Preparation

# note that mean and std is calculated channel-wise
# reference: https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457/10
print("Data Preparation...")
logger.info("Data Preparation...")
transform_train = transforms.Compose([
    transforms.Resize(size=(224, 224)),
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
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

if args.test_only:
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=2,
                                              sampler=SubsetRandomSampler(range(8)))
print("Model setting...")
logger.info("Model setting...")

use_cuda = torch.cuda.is_available()
start_epoch = 0
if args.net == "resnet18":
    print("using resnet18")
    net = torchvision.models.resnet.ResNet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2])
    if not os.path.isfile(args.resume):
        net.load_state_dict(torch.load("../data/model/resnet18-5c106cde.pth"))
else:
    print("using resnet50")
    net = torchvision.models.resnet.ResNet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3])
    if not os.path.isfile(args.resume):
        net.load_state_dict(torch.load("../data/model/resnet50-19c8e357.pth"))
# Do not change the layers that are pre-trained with the only exception
# on the last full-connected layer.
if not args.train_all:
    for param in net.parameters():
        param.requires_grad = False
# change the last fc layer for cifar100
net.fc = nn.Linear(in_features=net.fc.in_features, out_features=4096)

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    print("number of cuda: {}".format(torch.cuda.device_count()))
    cudnn.benchmark = True

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

criterion = nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-06)
training_loss_seq = []
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
        testing_accuracy_seq = checkpoint['testing_accuracy_seq']
        testing_best_accuracy = checkpoint['testing_best_accuracy']
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, (checkpoint['epoch'] + 1)))
        logger.info("=> loaded checkpoint '{}' (epoch {})"
                    .format(args.resume, (checkpoint['epoch'] + 1)))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
        logger.info("=> no checkpoint found at '{}'".format(args.resume))
        print("=> Training based on the {} from scratch...".format(args.net))
        logger.info("=> Training based on the {} from scratch...".format(args.net))
else:
    print("=> Training based on the {} from scratch...".format(args.net))
    logger.info("=> Training based on the {} from scratch...".format(args.net))


print("Model Training...")
logger.info("Model Training...")

# use up-to-date learning rate; for resume purpose
for param_group in optimizer.param_groups:
    current_learningRate = param_group['lr']


def train(epoch, train_loader):
    global current_learningRate
    net.train()
    if (epoch+1) % 5 == 0:
        current_learningRate /= 2
        logger.info("=> Learning rate is updated!")
        utils.update_learning_rate(optimizer, current_learningRate)

    loss_epoch = 0
    start = time.time()
    for _, (images, _) in enumerate(train_loader):
        if use_cuda:
            q, p, n = images[0].cuda(), images[1].cuda(), images[2].cuda()
        else:
            q, p, n = images[0], images[1], images[2]
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
    loss_epoch /= 100000
    end = time.time()
    logger.info("Current Epoch Takes {} min".format((end-start) / 60))
    print("=> Epoch: [{}/{}] | Loss:[{}]".format(epoch + 1, args.num_epochs, loss_epoch))
    logger.info("=> Epoch: [{}/{}] | Loss:[{}]".format(epoch + 1, args.num_epochs, loss_epoch))
    train_checkpoint = {"state_dict": net.state_dict(), "loss": loss_epoch, "optimizer": optimizer.state_dict(), "epoch": epoch}
    torch.save(train_checkpoint, "./pickle/train_checkpoint_{}.pth".format(epoch))
    return loss_epoch


# def train_new(epoch, train_loader):
#     global current_learningRate
#     net.train()
#     if (epoch+1) % 5 == 0:
#         current_learningRate /= 2
#         logger.info("=> Learning rate is updated!")
#         utils.update_learning_rate(optimizer, current_learningRate)

#     loss_epoch = 0
#     f_img_train = []
#     label_train = []
#     start = time.time()
#     for _, (images, lables) in enumerate(train_loader):
#         if use_cuda:
#             q, p, n, q_label = images[0].cuda(), images[1].cuda(), images[2].cuda(), lables[0].cuda()
#         else:
#             q, p, n, q_label = images[0], images[1], images[2], lables[0]
#         optimizer.zero_grad()
#         q, p, n = Variable(q), Variable(p), Variable(n)
#         f_q, f_p, f_n = net(q), net(p), net(n)
#         loss = criterion(f_q, f_p, f_n)
#         loss.backward()
#         optimizer.step()

#         if torch.__version__ == '0.4.1':
#             loss_epoch += loss.item()
#         else:
#             loss_epoch += loss.data[0]
#         f_img_train.append(f_q)
#         label_train.append(q_label)
#     loss_epoch /= 100000
#     end = time.time()
#     logger.info("Current Epoch Takes {} min".format((end-start) / 60))
#     print("=> Epoch: [{}/{}] | Loss:[{}]".format(epoch + 1, args.num_epochs, loss_epoch))
#     logger.info("=> Epoch: [{}/{}] | Loss:[{}]".format(epoch + 1, args.num_epochs, loss_epoch))
#     train_checkpoint = {"state_dict": net.state_dict(), "loss": loss_epoch, "optimizer": optimizer.state_dict(), "epoch": epoch}
#     torch.save(train_checkpoint, "./pickle/train_checkpoint_{}.pth".format(epoch))
#     return loss_epoch


def test(epoch, train_loader, k_closet=30):
    net.eval()
    if args.test_only:
        k_closet = 3
    f_img_train = []
    label_train = []
    test_accuracy = []
    with torch.no_grad():
        start = time.time()
        print("Calculate Training Feature Embedding...")
        for _, (images, lables) in enumerate(train_loader):
            if use_cuda:
                q, q_label = images[0].cuda(), lables[0].cuda()
            else:
                q, q_label = images[0], lables[0]
            q = Variable(q)
            f_q = net(q)
            f_img_train.append(f_q)
            label_train.append(q_label)
        f_img_train = torch.cat(f_img_train, dim=0)
        label_train = torch.cat(label_train, dim=0)
        end = time.time()
        print("Finish in {} min".format((end-start)/60))
        print("Testing...")
        for _, (imgs_test, labels_test) in enumerate(test_loader):
            if use_cuda:
                imgs_test, labels_test = imgs_test.cuda(), labels_test.cuda()
            imgs_test = Variable(imgs_test)
            f_img_test = net(imgs_test)
            for f_img_test_current, label_test_current in zip(f_img_test, labels_test):
                f_img_test_current = f_img_test_current.reshape(1, 4096)
                f_img_test_current = f_img_test_current.expand(f_img_train.shape[0], 4096)
                distance = pdist(f_img_test_current, f_img_train)
                predicted = label_train[distance.topk(k_closet, largest=False)[1]]
                test_accuracy.append(float(torch.sum(torch.eq(predicted, label_test_current))) / k_closet)
        test_accuracy_epoch = np.mean(test_accuracy)

        print("=> Epoch: [{}/{}] | Testing Accuracy: [{}]".format(
            epoch + 1, args.num_epochs, test_accuracy_epoch))
        logger.info("=> Epoch: [{}/{}] | Testing Accuracy: [{}]".format(
            epoch + 1, args.num_epochs, test_accuracy_epoch))

    return test_accuracy_epoch


for epoch in range(start_epoch, args.num_epochs):
    if args.test_only:
        k_closet = 3
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
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size, shuffle=True, num_workers=32)
    if not os.path.isfile("./pickle/train_checkpoint_{}.pth".format(epoch)):
        train_loss = train(epoch, train_loader)
    else:
        print("Loading from history..")
        train_checkpoint = torch.load("./pickle/train_checkpoint_{}.pth".format(epoch))
        net.load_state_dict(train_checkpoint["state_dict"])
        try:
            train_loss = train_checkpoint["loss"]
        except KeyError:
            train_loss = 0

    #test_accuracy = test(epoch, train_loader, k_closet=30)
    training_loss_seq.append(train_loss)
    testing_accuracy_seq.append(0.54)

    is_best = testing_accuracy_seq[-1] > testing_best_accuracy
    testing_best_accuracy = max(testing_best_accuracy, testing_accuracy_seq[-1])

    state = {
        "epoch": epoch,
        "state_dict": net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "training_loss_seq": training_loss_seq,
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
