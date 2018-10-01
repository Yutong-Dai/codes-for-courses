import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
import h5py
import copy
import time
import torch.backends.cudnn as cudnn

# print(torch.__version__)
file_name = "../data/CIFAR10.hdf5"
data = h5py.File(file_name, "r")
# get metadata for data: [n for n in data.keys()]
x_train = np.float32(data["X_train"][:]).reshape(-1, 3, 32, 32)
y_train = np.int32(np.array(data["Y_train"]))
x_test = np.float32(data["X_test"][:]).reshape(-1, 3, 32, 32)
y_test = np.int32(np.array(data["Y_test"]))
data.close()
print("data loaded!")

class CNN_torch(nn.Module):
    def __init__(self):
        super(CNN_torch, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=1, padding=2, bias=True)
        self.batch_norm1 = nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=2, bias=True)
        self.dropout1 = nn.Dropout2d(p=0.25)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=2, bias=True)
        self.batch_norm2 = nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=2, bias=True)
        self.dropout2 = nn.Dropout2d(p=0.25)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=2, bias=True)
        self.batch_norm3 = nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, bias=True)
        self.dropout3 = nn.Dropout2d(p=0.25)
        self.conv7 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, bias=True)
        self.batch_norm4 = nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True)
        self.conv8 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, bias=True)
        self.dropout4 = nn.Dropout2d(p=0.25)
        self.batch_norm5 = nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True)
        self.fc1 = nn.Linear(64 * 4 * 4, 500)
        self.fc2 = nn.Linear(500, 10)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.batch_norm1(x)
        x = F.relu(self.conv2(x))
        x = self.dropout1(F.max_pool2d(x, kernel_size=2, stride=2))
        x = F.relu(self.conv3(x))
        x = self.batch_norm2(x)
        x = F.relu(self.conv4(x))
        x = self.dropout2(F.max_pool2d(x, kernel_size=2, stride=2))
        x = F.relu(self.conv5(x))
        x = self.batch_norm3(x)
        x = F.relu(self.conv6(x))
        x = self.dropout3(x)
        x = F.relu(self.conv7(x))
        x = self.batch_norm4(x)
        x = F.relu(self.conv8(x))
        x = self.batch_norm5(x)
        x = self.dropout4(x)
        x = x.view(-1, self.fc1.in_features)
        x = self.fc1(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
model = CNN_torch()
use_cuda = torch.cuda.is_available()
if use_cuda:
    model.cuda()
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    print(torch.cuda.device_count())
    cudnn.benchmark = True
resume = './checkpoint.pth.tar'
checkpoint = torch.load(resume)
start_epoch = checkpoint['epoch']
best_train_acc = checkpoint['best_train_acc']
train_loss = checkpoint['train_loss']
train_accuracy_epoch = checkpoint['train_accuracy_epoch']
model.load_state_dict(checkpoint['state_dict'])

print("model loaded!")
use_cuda = torch.cuda.is_available()
batch_size = 8
model.train()
test_accuracy = []
for i in range(0, len(y_test), batch_size):
    x_test_batch = torch.FloatTensor(x_test[i:i+batch_size, :])
    y_test_batch = torch.LongTensor(y_test[i:i+batch_size])
    if use_cuda:
        data, target = Variable(x_test_batch).cuda(), Variable(y_test_batch).cuda()
    else:
        data, target = Variable(x_test_batch), Variable(y_test_batch)
    output = model.forward(data)
    mc_output = output
    for i in range(49):
        output = model.forward(data)
        mc_output += output
    mc_output = mc_output / 50
    prediction = mc_output.data.max(1)[1]
    accuracy = (float(prediction.eq(target.data).sum()) / float(batch_size))
    test_accuracy.append(accuracy)
accuracy_test = np.mean(test_accuracy)
print(accuracy_test)

# 0.8108