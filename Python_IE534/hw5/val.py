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
from collections import OrderedDict

parser = argparse.ArgumentParser(description='PyTorch TinyImageNet Training 200 classes')
parser.add_argument('--epoch', '-e', default="14", help='epoch to use')
args = parser.parse_args()

pdist = nn.PairwiseDistance(p=2)
epoch = args.epoch  # load idx
net = torchvision.models.resnet.ResNet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2])
net.fc = nn.Linear(in_features=net.fc.in_features, out_features=4096)
use_cuda = torch.cuda.is_available()
if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    print("number of cuda: {}".format(torch.cuda.device_count()))
    cudnn.benchmark = True

train_checkpoint = torch.load("./pickle/train_checkpoint_{}.pth".format(epoch))
state_dict = train_checkpoint["state_dict"]
net.load_state_dict(state_dict)

print("Data Preparation...")
transform_test = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


f_img_train, label_train = pickle.load(open("./pickle/embedding_{}.p".format(epoch), 'rb'))
print("loading from the history!")

img, label = utils.generate_testing_data_set()
val_img = [img[i] for i in [2, 4, 8, 9, 12]]
val_label = [label[i] for i in [2, 4, 8, 9, 12]]
test_dataset = utils.TinyImageNet(val_img, val_label, train=False, transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=32)
top10 = {"idx_of_fig_path_top10": [], "distance_of_top10": [], "predicted_top10": []}
bottom10 = {"idx_of_fig_path_bottom10": [], "distance_of_bottom10": [], "predicted_bottom10": []}

print("begin validation!")
with torch.no_grad():
    for _, (imgs_test, labels_test) in enumerate(test_loader):
        if use_cuda:
            imgs_test, labels_test = imgs_test.cuda(), labels_test.cuda()
        imgs_test = Variable(imgs_test)
        f_img_test = net(imgs_test)
        for f_img_test_current, label_test_current in zip(f_img_test, labels_test):
            f_img_test_current = f_img_test_current.reshape(1, 4096)
            f_img_test_current = f_img_test_current.expand(f_img_train.shape[0], 4096)
            distance = pdist(f_img_test_current, f_img_train)

            idx_top10 = distance.topk(10, largest=False)[1]
            top10["idx_of_fig_path_top10"].append(idx_top10)
            top10["predicted_top10"].append(label_train[idx_top10])
            top10["distance_of_top10"].append(distance.topk(10, largest=False)[0])

            idx_bottom10 = distance.topk(10, largest=True)[1]
            bottom10["idx_of_fig_path_bottom10"].append(idx_bottom10)
            bottom10["predicted_bottom10"].append(label_train[idx_bottom10])
            bottom10["distance_of_bottom10"].append(distance.topk(10, largest=True)[0])

torch.save((top10, bottom10), "./final.pth")
print("saved!")
