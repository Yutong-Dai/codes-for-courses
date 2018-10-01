import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import os
import argparse
import logging
import shutil
import numpy as np

from myutils import *
from resnet import *

# General setups
parser = argparse.ArgumentParser(description='PyTorch CIFAR-100 Training')
parser.add_argument('--resume', '-r', default="./checkpoint.pth.tar", help='resume from checkpoint')
parser.add_argument('--epoch', '-e', default=30, help='total training epoch')
args = parser.parse_args()

# python hw4_bw.py  --resume --epoch 50

if args.log:
    log_level=logging.INFO
    logger = logging.getLogger()
    logger.setLevel(log_level)
    handler = logging.FileHandler("task.log")
    handler.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.info("torch version: {}".format(torch.__version__))


# Hyper Parameters
use_cuda = torch.cuda.is_available()
best_acc = 0
start_epoch, num_epochs, batch_size = 1, 30, 256

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
num_classes = 100

print("Loading Data...")
logger.info("Loading Data...")
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64,shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)

print("Model setting...")

if args.