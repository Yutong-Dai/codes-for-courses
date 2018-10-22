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


parser = argparse.ArgumentParser(description='PyTorch TinyImageNet Training 200 classes')
parser.add_argument('--epoch', '-e', default="14", type=int, help='epoch to use')
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


img_triplet, label_triplet = pickle.load(open("./pickle/train_{}.p".format(epoch), 'rb'))
train_dataset = utils.TinyImageNet(img_triplet, label_triplet, transform=transform_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, num_workers=32, shuffle=False)

img, label = utils.generate_testing_data_set()
test_dataset = utils.TinyImageNet(img, label, train=False, transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False, num_workers=32)


def test(epoch, train_loader, k_closet=30, use_cuda=True):
    net.eval()
    f_img_train = []
    label_train = []
    test_accuracy = []
    with torch.no_grad():
        if not os.path.isfile("./pickle/embedding_{}.p".format(epoch)):
            print("Calculate Training Feature Embedding...")
            start = time.time()
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
            pickle.dump((f_img_train, label_train), file=open("./pickle/embedding_{}.p".format(epoch), "wb"))
            print("saved!")
        else:
            f_img_train, label_train = pickle.load(open("./pickle/embedding_{}.p".format(epoch), 'rb'))
            print("loaded!")
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
            print(test_accuracy[-1])
        test_accuracy_epoch = np.mean(test_accuracy)

        print("=> Epoch: [{}] | Testing Accuracy: [{}]".format(epoch + 1,  test_accuracy_epoch))
    return test_accuracy_epoch


test_accuracy = test(epoch, train_loader, k_closet=30)

torch.save(test_accuracy, "./pickle/test_accuracy_{}.pth".format(epoch))
